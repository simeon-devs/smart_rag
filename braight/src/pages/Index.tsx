import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import type { Product, ChatMessage, ConstraintSuggestion } from "@/types/product";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { useLanguage } from "@/contexts/LanguageContext";
import { useWishlists } from "@/hooks/useWishlists";
import { useProjects } from "@/hooks/useProjects";
import { useTracking } from "@/hooks/useTracking";
import { extractConstraints, saveConstraints, chatWithMara, browseMara } from "@/lib/maraApi";
import Header from "@/components/Header";
import WelcomeState from "@/components/WelcomeState";
import ArcCarousel from "@/components/ArcCarousel";
import MobileProducts from "@/components/MobileProducts";
import PitMascot from "@/components/PitMascot";
import ChatWindow from "@/components/ChatWindow";
import ProductDetail from "@/components/ProductDetail";
import WishlistOverlay from "@/components/WishlistOverlay";
import ProjectOverlay from "@/components/ProjectOverlay";
import AuthSplash from "@/components/AuthSplash";
import type { WishlistWithItems } from "@/hooks/useWishlists";
import type { ProjectWithItems } from "@/hooks/useProjects";

function getGuestSessionId(): string {
  const KEY = 'mara_sid';
  let sid = sessionStorage.getItem(KEY);
  if (!sid) {
    sid = `guest_${crypto.randomUUID()}`;
    sessionStorage.setItem(KEY, sid);
  }
  return sid;
}

const Index = () => {
  const { user, profile, loading: authLoading } = useAuth();
  const maraUserId = useMemo(() => user?.id ?? getGuestSessionId(), [user?.id]);
  const { t, locale } = useLanguage();
  const [visibleProducts, setVisibleProducts] = useState<Product[]>([]);
  const [allProducts, setAllProducts] = useState<Product[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [activeWishlist, setActiveWishlist] = useState<WishlistWithItems | null>(null);
  const [activeProject, setActiveProject] = useState<ProjectWithItems | null>(null);
  const [pitMode, setPitMode] = useState<'idle' | 'thinking' | 'excited'>('idle');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: t('chat_initial') }
  ]);
  const [isThinking, setIsThinking] = useState(false);
  const [showAuthSplash, setShowAuthSplash] = useState(false);
  const hasSearchedRef = useRef(false);
  const lastMessageRef = useRef<string>('');

  const wl = useWishlists();
  const proj = useProjects();
  const { track, trackInteraction } = useTracking();
  const lastQueryRef = useRef<string>('');

  const showWelcome = visibleProducts.length === 0;

  useEffect(() => {
    supabase.from('articles').select('id, l_number, article_number, very_short_description_de, short_description_de, long_description_de, hero_image_url, article_classification_id, article_character_profile_id, article_technical_profile_id, light_category_id, light_family_id, manufacturer_id, is_current, price_sp_chf, price_pp_chf').eq('is_current', true).order('id').limit(200).then(({ data }) => {
      if (data) setAllProducts(data as unknown as Product[]);
    });
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setSelectedProduct(null); setActiveWishlist(null); setActiveProject(null); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const hydrateFromSupabase = useCallback(async (articleIds: number[]): Promise<Product[]> => {
    if (articleIds.length === 0) return [];
    const { data, error } = await supabase
      .from('articles')
      .select('id, l_number, article_number, very_short_description_de, short_description_de, long_description_de, hero_image_url, article_classification_id, article_character_profile_id, article_technical_profile_id, light_category_id, light_family_id, manufacturer_id, is_current, price_sp_chf, price_pp_chf')
      .in('id', articleIds).eq('is_current', true);
    if (error) return [];
    const byId = new Map((data || []).map(d => [d.id, d]));
    return articleIds.map(id => byId.get(id)).filter(Boolean) as unknown as Product[];
  }, []);

  const handleSendMessage = useCallback(async (text: string) => {
    track('search_query', { query: text });
    lastQueryRef.current = text;
    setIsThinking(true); setPitMode('thinking');
    setChatMessages(prev => [...prev, { role: 'user', content: text }]);
    lastMessageRef.current = text;
    try {
      let suggestions: ConstraintSuggestion[] = [];
      try { const extractRes = await extractConstraints(maraUserId, text); suggestions = extractRes.suggestions || []; } catch {}
      const chatRes = await chatWithMara(maraUserId, text);
      let articleIds = chatRes.hydration?.ordered_article_ids || [];
      // Fallback: if no ordered_article_ids, use baseline_results or mara_results
      if (articleIds.length === 0) {
        const fallbackResults = (chatRes.mara_results?.length ? chatRes.mara_results : chatRes.baseline_results) || [];
        articleIds = fallbackResults.map((r: any) => r.source_article_id).filter(Boolean);
      }
      const hydratedProducts = await hydrateFromSupabase(articleIds);
      const allResults = [...(chatRes.mara_results || []), ...(chatRes.baseline_results || [])];
      const maraMap = new Map(allResults.map((r: any) => [r.source_article_id, r]));
      const enriched = hydratedProducts.map(p => {
        const mara = maraMap.get(p.id);
        return { ...p, mara_score: mara?.final_score ?? mara?.similarity_score, mara_violations: mara?.violations };
      });
      setChatMessages(prev => [...prev, { role: 'assistant', content: chatRes.llm_reply, suggestions: suggestions.length > 0 ? suggestions : undefined }]);
      // Log search to search_logs
      try {
        const sessionId = sessionStorage.getItem("_sid") || '';
        const maraScores = (chatRes.mara_results || []).map((r: any) => ({
          article_id: r.source_article_id,
          score: r.final_score,
          violations: r.violations,
        }));
        await supabase.from("product_interactions").insert({
          user_id: user?.id ?? null,
          session_id: sessionId,
          article_id: null,
          interaction_type: 'search',
          search_query: text,
          llm_reply: chatRes.llm_reply,
          returned_article_ids: articleIds,
          mara_scores: maraScores,
          constraint_suggestions: suggestions,
        } as any);
      } catch { /* silent */ }
      setIsThinking(false); setPitMode('excited'); hasSearchedRef.current = true;
      if (!user && !authLoading) { setTimeout(() => { setShowAuthSplash(true); setPitMode('idle'); }, 600); }
      else { setTimeout(() => setVisibleProducts(enriched), 400); setTimeout(() => setPitMode('idle'), 2500); }
    } catch (err) {
      console.error('[FLOW] Chat error:', err);
      setChatMessages(prev => [...prev, { role: 'assistant', content: t('chat_error') }]);
      setIsThinking(false); setPitMode('idle');
    }
  }, [user, authLoading, maraUserId, track, hydrateFromSupabase, t]);

  const handleConstraintAction = useCallback(async (suggestion: ConstraintSuggestion, accepted: boolean) => {
    // Track constraint accept/reject
    trackInteraction(0, accepted ? 'constraint_accept' : 'constraint_reject', {
      searchQuery: lastQueryRef.current || undefined,
      extra: { field: suggestion.field, label: suggestion.label, value: suggestion.value },
    });
    if (!accepted) { setChatMessages(prev => prev.map(m => ({ ...m, suggestions: m.suggestions?.filter(s => s.field !== suggestion.field) }))); return; }
    try {
      await saveConstraints(maraUserId, { [suggestion.field]: suggestion.value });
      setChatMessages(prev => prev.map(m => ({ ...m, suggestions: m.suggestions?.filter(s => s.field !== suggestion.field) })));
      setChatMessages(prev => [...prev, { role: 'assistant', content: t('constraint_set', { label: suggestion.label }) }]);
    } catch {}
  }, [maraUserId, trackInteraction, t]);

  const isProfileComplete = profile && profile.company_name;

  useEffect(() => {
    if (user && isProfileComplete && hasSearchedRef.current && visibleProducts.length === 0) {
      setShowAuthSplash(false);
      const lastUserMsg = [...chatMessages].reverse().find(m => m.role === 'user');
      if (lastUserMsg) handleSendMessage(lastUserMsg.content);
    }
  }, [user, isProfileComplete]);

  const productOpenTimeRef = useRef<number>(0);

  const handleProductClick = useCallback((product: Product, carouselPosition?: number) => {
    setSelectedProduct(product);
    productOpenTimeRef.current = Date.now();
    track('product_view', { dwell_start: Date.now() }, product.id);
    trackInteraction(product.id, 'product_view', {
      searchQuery: lastQueryRef.current || undefined,
      carouselPosition,
    });
    // Save browse event as MARA episodic memory (fire-and-forget)
    browseMara(
      maraUserId,
      `article_${product.id}`,
      product.very_short_description_de || product.article_number,
      product.short_description_de || '',
    ).catch(() => { /* silent — memory is best-effort */ });
  }, [maraUserId, track, trackInteraction]);

  const handleProductClose = useCallback(() => {
    if (selectedProduct && productOpenTimeRef.current) {
      const dwellMs = Date.now() - productOpenTimeRef.current;
      trackInteraction(selectedProduct.id, 'product_close', {
        dwellMs,
        searchQuery: lastQueryRef.current || undefined,
      });
    }
    setSelectedProduct(null);
  }, [selectedProduct, trackInteraction]);

  const requireAuth = useCallback(() => {
    if (!user) { setShowAuthSplash(true); return false; }
    return true;
  }, [user]);

  const handleAddToProject = useCallback((projectId: number, articleId: number, qty = 1) => {
    if (!requireAuth()) return;
    proj.addToProject(projectId, articleId, qty);
    track('add_to_cart', { quantity: qty, projectId }, articleId);
    const projectName = proj.projects.find(p => p.id === projectId)?.project_name;
    trackInteraction(articleId, 'project_add', {
      searchQuery: lastQueryRef.current || undefined,
      projectName: projectName || undefined,
      quantity: qty,
    });
  }, [requireAuth, proj, track, trackInteraction]);

  const handleToggleWishlist = useCallback((wishlistId: number, articleId: number, isIn: boolean) => {
    if (!requireAuth()) return;
    const wishlistName = wl.wishlists.find(w => w.id === wishlistId)?.name;
    if (isIn) {
      wl.removeFromWishlist(wishlistId, articleId);
      trackInteraction(articleId, 'wishlist_remove', { wishlistName, searchQuery: lastQueryRef.current || undefined });
    } else {
      wl.addToWishlist(wishlistId, articleId);
      trackInteraction(articleId, 'wishlist_add', { wishlistName, searchQuery: lastQueryRef.current || undefined });
    }
  }, [requireAuth, wl, trackInteraction]);

  const handleReject = useCallback((articleId: number) => {
    if (!requireAuth()) return;
    wl.rejectArticle(articleId);
    track('product_reject', {}, articleId);
    trackInteraction(articleId, 'product_reject', { searchQuery: lastQueryRef.current || undefined });
  }, [requireAuth, wl, track, trackInteraction]);

  const handleCreateWishlist = useCallback(() => { if (requireAuth()) wl.createWishlist(); }, [requireAuth, wl]);
  const handleCreateProject = useCallback(() => { if (requireAuth()) proj.createProject(); }, [requireAuth, proj]);

  const handleRequestDelivery = useCallback(async (articleIds: number[]) => {
    if (!requireAuth() || articleIds.length === 0) return;
    // Create a new project from wishlist items and submit
    const dateLocale = locale === 'de' ? 'de-CH' : locale === 'fr' ? 'fr-CH' : locale === 'it' ? 'it-CH' : 'en-GB';
    const { data: newProj } = await supabase.from("projects").insert({
      user_id: user!.id,
      project_name: t('inquiry_date', { date: new Date().toLocaleDateString(dateLocale) }),
      status: "submitted" as any,
    }).select().single();
    if (!newProj) return;
    for (const aid of articleIds) {
      const product = allProducts.find(p => p.id === aid);
      await supabase.from("project_items").insert({
        project_id: newProj.id,
        article_id: aid,
        quantity: 1,
        unit_price_chf: product?.price_sp_chf ? parseFloat(product.price_sp_chf) : null,
      });
    }
    track('delivery_request', { project_id: newProj.id });
    for (const aid of articleIds) {
      trackInteraction(aid, 'delivery_request', { searchQuery: lastQueryRef.current || undefined });
    }
    setChatMessages(prev => [...prev, { role: 'assistant', content: t('inquiry_created', { id: String(newProj.id) }) }]);
    proj.loadProjects();
  }, [requireAuth, user, allProducts, track, trackInteraction, proj, t, locale]);

  // Keep activeProject synced with latest data
  useEffect(() => {
    if (activeProject) {
      const updated = proj.projects.find(p => p.id === activeProject.id);
      if (updated) setActiveProject(updated);
    }
  }, [proj.projects]);

  // Keep activeWishlist synced
  useEffect(() => {
    if (activeWishlist) {
      const updated = wl.wishlists.find(w => w.id === activeWishlist.id);
      if (updated) setActiveWishlist(updated);
    }
  }, [wl.wishlists]);

  const displayProducts = visibleProducts.filter(p => !wl.rejectedIds.has(p.id));

  return (
    <div className="relative w-full h-screen overflow-hidden">
      <Header
        wishlists={wl.wishlists}
        projects={proj.projects}
        onOpenWishlist={w => setActiveWishlist(w)}
        onCreateWishlist={handleCreateWishlist}
        onOpenProject={p => setActiveProject(p)}
        onCreateProject={handleCreateProject}
        getProjectColor={proj.getColor}
      />

      <WelcomeState visible={showWelcome} />

      <ArcCarousel
        products={displayProducts} wishlists={wl.wishlists}
        projects={proj.projects} getProjectColor={proj.getColor}
        rejectedIds={wl.rejectedIds}
        onToggleWishlist={handleToggleWishlist} onReject={handleReject}
        onAddToProject={handleAddToProject} onCreateProject={handleCreateProject}
        onRequestDelivery={handleRequestDelivery}
        onProductClick={handleProductClick}
      />

      <MobileProducts products={displayProducts} onProductClick={handleProductClick} />
      <PitMascot mode={pitMode} />

      <ChatWindow messages={chatMessages} isThinking={isThinking} onSendMessage={handleSendMessage} onConstraintAction={handleConstraintAction} />

      <ProductDetail
        product={selectedProduct}
        isSaved={selectedProduct ? wl.getWishlistsForArticle(selectedProduct.id).length > 0 : false}
        onToggleSave={(id) => {
          if (wl.wishlists.length > 0) {
            const first = wl.wishlists[0];
            handleToggleWishlist(first.id, id, first.article_ids.includes(id));
          }
        }}
        onClose={handleProductClose}
        onAddToCart={(p) => {
          if (proj.projects.length > 0) handleAddToProject(proj.projects[0].id, p.id, 1);
        }}
      />

      <WishlistOverlay
        open={!!activeWishlist} wishlist={activeWishlist} allWishlists={wl.wishlists}
        allProducts={allProducts}
        projects={proj.projects} getProjectColor={proj.getColor}
        onClose={() => setActiveWishlist(null)}
        onRemoveFromWishlist={wl.removeFromWishlist} onMoveToWishlist={wl.moveToWishlist}
        onAddToProject={handleAddToProject} onCreateProject={handleCreateProject}
        onRename={wl.renameWishlist} onDelete={wl.deleteWishlist}
        onRequestDelivery={handleRequestDelivery}
      />

      <ProjectOverlay
        open={!!activeProject} project={activeProject}
        color={activeProject ? proj.getColor(proj.projects.indexOf(activeProject)) : '#C8932A'}
        allProducts={allProducts} onClose={() => setActiveProject(null)}
        onRename={proj.renameProject} onDelete={proj.deleteProject}
        onUpdateQuantity={proj.updateItemQuantity} onRemoveItem={proj.removeItem}
        onSubmit={(id) => {
          proj.submitProject(id);
          const project = proj.projects.find(p => p.id === id);
          // Track each article in the submitted project
          project?.items.forEach(item => {
            trackInteraction(item.article_id, 'project_submit', {
              projectName: project.project_name || undefined,
              quantity: item.quantity,
            });
          });
          setChatMessages(prev => [...prev, { role: 'assistant', content: t('inquiry_submitted') }]);
        }}
      />

      <AuthSplash open={showAuthSplash} onClose={() => setShowAuthSplash(false)} />

      {displayProducts.length > 4 && (
        <div className="fixed bottom-[22px] left-1/2 -translate-x-1/2 text-[10px] tracking-[0.18em] uppercase text-muted-foreground z-20 whitespace-nowrap animate-pulse">
          {t('scroll_to_explore')}
        </div>
      )}
    </div>
  );
};

export default Index;

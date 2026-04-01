# 1. The product type definition — tells us exact data structure
cat braight/src/types/product.ts

# 2. The Supabase types — tells us exact database columns
cat braight/src/integrations/supabase/types.ts

# 3. The chat function — tells us how Pit currently calls AI
cat braight/supabase/functions/chat-recommend/index.ts

# 4. The first migration — tells us how the database was created
cat braight/supabase/migrations/20260308075200_7531f97b-a962-4757-a814-1b4d7b842a87.sql
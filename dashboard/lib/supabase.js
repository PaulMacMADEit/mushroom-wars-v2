// Supabase REST client for the static dashboard.
//
// Loaded via ESM CDN so there's no build step — any page just imports from here.
// The anon key is safe to publish: RLS policies (infra/rls.sql) restrict it to
// SELECT-only.

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const SUPABASE_URL  = 'https://lwkljcyspyqklyoagnmo.supabase.co';
const SUPABASE_ANON = 'sb_publishable_S7q8SQrOn6W4OJ7tSFqJWQ_HFrf63KU';

export const sb = createClient(SUPABASE_URL, SUPABASE_ANON);

/** Fetch a short-lived signed URL for an artifact path like "models/abc/weights.pt". */
export async function signedUrl(path, expiresIn = 3600) {
  if (!path) return null;
  const slash = path.indexOf('/');
  if (slash < 0) throw new Error(`bad storage path: ${path}`);
  const bucket = path.slice(0, slash);
  const key    = path.slice(slash + 1);
  const { data, error } = await sb.storage.from(bucket).createSignedUrl(key, expiresIn);
  if (error) throw error;
  return data.signedUrl;
}

/** Read the current URL's ?id=... (for per-entity detail pages). */
export function queryParam(name) {
  return new URLSearchParams(window.location.search).get(name);
}

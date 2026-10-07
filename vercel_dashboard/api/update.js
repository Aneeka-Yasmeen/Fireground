import { put } from '@vercel/blob';
import crypto from 'node:crypto';

function authorized(request) {
  const expected = process.env.FIREGROUND_INGEST_TOKEN;
  const header = request.headers.get('authorization') || '';
  if (!expected || !header.startsWith('Bearer ')) return false;

  const provided = header.slice('Bearer '.length);
  const a = Buffer.from(provided);
  const b = Buffer.from(expected);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

export async function POST(request) {
  if (!authorized(request)) {
    return Response.json({ ok: false, error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const raw = await request.text();
    if (raw.length > 2_000_000) {
      return Response.json({ ok: false, error: 'Payload too large' }, { status: 413 });
    }

    const data = JSON.parse(raw);
    if (!data || !Array.isArray(data.results)) {
      return Response.json({ ok: false, error: 'Invalid Fireground result payload' }, { status: 400 });
    }

    await put(
      'fireground/latest.json',
      JSON.stringify(data),
      {
        access: 'private',
        allowOverwrite: true,
        contentType: 'application/json',
      },
    );

    return Response.json({
      ok: true,
      windows: data.results.length,
      overall_status: data.overall_status ?? null,
      updated_at: new Date().toISOString(),
    });
  } catch (error) {
    console.error('update API error', error);
    return Response.json(
      { ok: false, error: 'Invalid JSON or storage failure' },
      { status: 400 },
    );
  }
}

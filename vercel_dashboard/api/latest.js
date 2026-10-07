import { get } from '@vercel/blob';

export async function GET(request) {
  try {
    const result = await get('fireground/latest.json', {
      access: 'private',
      useCache: false,
    });

    if (!result || result.statusCode !== 200) {
      return Response.json(
        { ok: false, error: 'No live VNNX result has been published yet.' },
        { status: 404, headers: { 'Cache-Control': 'no-store' } },
      );
    }

    const text = await new Response(result.stream).text();

    return new Response(text, {
      status: 200,
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
        'Cache-Control': 'private, no-store',
        'X-Content-Type-Options': 'nosniff',
      },
    });
  } catch (error) {
    console.error('latest API error', error);
    return Response.json(
      { ok: false, error: 'Unable to read live VNNX data.' },
      { status: 500, headers: { 'Cache-Control': 'no-store' } },
    );
  }
}

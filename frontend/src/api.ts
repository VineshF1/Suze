const BASE_URL = '/api/v1';

export async function chat(query: string): Promise<Response> {
  return fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
}

export async function ingestText(content: string, source: string, metadata: Record<string, any> = {}) {
  const res = await fetch(`${BASE_URL}/ingest/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, source, metadata }),
  });
  return res.json();
}

export async function ingestFile(file: File) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE_URL}/ingest/file`, {
    method: 'POST',
    body: form,
  });
  return res.json();
}

export async function healthCheck() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}

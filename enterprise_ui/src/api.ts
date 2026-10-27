import type {
  DocumentRecord,
  ExtractionRecord,
  StreamEvent,
  Workspace,
} from "./types";

const UUID =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function requireUuid(value: string): string {
  if (!UUID.test(value)) throw new ApiError(0, "Invalid identifier");
  return value;
}

async function detail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail.slice(0, 500);
  } catch {
    /* not JSON */
  }
  return "";
}

export async function readNdjson(
  response: Response,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  if (!response.body) throw new ApiError(response.status, "Empty stream");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const emit = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      onEvent(JSON.parse(trimmed) as StreamEvent);
    } catch {
      throw new ApiError(response.status, "The stream ended before a complete answer was released.");
    }
  };
  try {
    for (;;) {
      const step = await reader.read();
      if (step.done) break;
      buffer += decoder.decode(step.value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) emit(line);
    }
    buffer += decoder.decode();
    if (buffer.trim()) emit(buffer);
  } finally {
    reader.releaseLock();
  }
}

export function createClient(token: string, fetchImpl: typeof fetch = fetch) {
  async function send(path: string, init: RequestInit = {}): Promise<Response> {
    const headers = new Headers(init.headers);
    headers.set("Authorization", `Bearer ${token}`);
    let response: Response;
    try {
      response = await fetchImpl(path, { ...init, headers });
    } catch {
      throw new ApiError(0, "");
    }
    return response;
  }

  async function json<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await send(path, init);
    if (!response.ok) throw new ApiError(response.status, await detail(response));
    return (await response.json()) as T;
  }

  return {
    async workspaces(): Promise<Workspace[]> {
      const body = await json<{ workspaces: Workspace[] }>("/api/workspaces");
      return body.workspaces.map((row) => ({ id: String(row.id), name: String(row.name) }));
    },
    async documents(workspaceId: string): Promise<DocumentRecord[]> {
      const id = requireUuid(workspaceId);
      const body = await json<{ documents: DocumentRecord[] }>(
        `/api/workspaces/${id}/documents`,
      );
      return body.documents.map((row) => ({
        ...row,
        id: String(row.id),
        version_id: String(row.version_id),
        filename: String(row.filename),
      }));
    },
    async extraction(workspaceId: string, versionId: string): Promise<ExtractionRecord> {
      return json(
        `/api/workspaces/${requireUuid(workspaceId)}/versions/${requireUuid(versionId)}/extraction`,
      );
    },
    async upload(workspaceId: string, file: File): Promise<{ version_id: string; filename: string }> {
      const response = await send(`/api/workspaces/${requireUuid(workspaceId)}/documents`, {
        method: "POST",
        headers: { "X-Filename": encodeURIComponent(file.name) },
        body: file,
      });
      if (!response.ok) throw new ApiError(response.status, await detail(response));
      const body = (await response.json()) as { version_id: string; filename: string };
      return { version_id: String(body.version_id), filename: String(body.filename) };
    },
    async cancelExtraction(workspaceId: string, versionId: string): Promise<void> {
      const response = await send(
        `/api/workspaces/${requireUuid(workspaceId)}/versions/${requireUuid(versionId)}/cancel`,
        { method: "POST" },
      );
      if (!response.ok) throw new ApiError(response.status, await detail(response));
    },
    async openStream(
      workspaceId: string,
      kind: "answers" | "coverage",
      payload: { question: string; version_ids: string[]; client_request_id: string },
      signal: AbortSignal,
    ): Promise<Response> {
      const response = await send(`/api/workspaces/${requireUuid(workspaceId)}/${kind}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal,
      });
      if (!response.ok) throw new ApiError(response.status, await detail(response));
      return response;
    },
    async cancelRun(workspaceId: string, kind: "answers" | "coverage", runId: string): Promise<void> {
      const response = await send(
        `/api/workspaces/${requireUuid(workspaceId)}/${kind}/${requireUuid(runId)}/cancel`,
        { method: "POST" },
      );
      if (!response.ok && response.status !== 404) {
        throw new ApiError(response.status, await detail(response));
      }
    },
    async coverageMap(workspaceId: string, runId: string): Promise<Record<string, unknown>> {
      return json(
        `/api/workspaces/${requireUuid(workspaceId)}/coverage/${requireUuid(runId)}`,
      );
    },
    async preview(path: string): Promise<string> {
      const response = await send(path);
      if (!response.ok) throw new ApiError(response.status, await detail(response));
      return response.text();
    },
  };
}

export type EnterpriseClient = ReturnType<typeof createClient>;

export const MAX_UPLOAD_BYTES = 5_000_000;
const ALLOWED = new Set(["pdf", "txt", "md", "csv"]);

export function uploadProblem(file: { name: string; size: number }): "type" | "size" | null {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!ALLOWED.has(extension) || file.name.length > 240 || /[\\/]/.test(file.name)) return "type";
  if (file.size <= 0 || file.size > MAX_UPLOAD_BYTES) return "size";
  return null;
}

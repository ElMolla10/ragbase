import { useEffect, useRef, useState } from "react";
import { ApiError, createClient, readNdjson, uploadProblem } from "./api";
import {
  calculationDownloadPath,
  calculationId,
  isCsvFilename,
  readCalculationRecord,
  readDownloads,
  stripCalculate,
} from "./calculations/record";
import { previewPath } from "./disclosure";
import { useLocale } from "./i18n";
import type {
  AnswerPayload,
  CalculationResponse,
  ExtractionRecord,
  HealthGeneration,
  HealthReady,
  StreamEvent,
  Workspace,
} from "./types";
import {
  ChatPanel,
  DocumentPanel,
  HealthStrip,
  LanguageToggle,
  SignIn,
  SourcePreview,
  type ChatEntry,
  type DocItem,
} from "./views";

const TOKEN_KEY = "ragbase.enterprise.devToken";

function toDoc(row: {
  id: string;
  filename: string;
  version_id: string;
  version_number: number;
  byte_size: number;
  sha256: string;
  ingestion_status: string;
}): DocItem {
  return {
    id: row.id,
    filename: row.filename,
    versionId: row.version_id,
    versionNumber: row.version_number,
    byteSize: row.byte_size,
    sha256: row.sha256,
    ingestionStatus: row.ingestion_status,
    retrievalReady: null,
    warnings: [],
    errorDetail: null,
  };
}

function withExtraction(doc: DocItem, extraction: ExtractionRecord): DocItem {
  return {
    ...doc,
    ingestionStatus: extraction.status || doc.ingestionStatus,
    retrievalReady: extraction.retrieval_ready === true,
    warnings: extraction.warnings ?? [],
    errorDetail: extraction.error_detail ?? null,
  };
}

export function App() {
  const { t } = useLocale();
  const [token, setToken] = useState<string | null>(() => {
    try {
      return sessionStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  });
  const [authError, setAuthError] = useState<string | null>(null);
  const [authBusy, setAuthBusy] = useState(false);
  const [ready, setReady] = useState<HealthReady | null>(null);
  const [generation, setGeneration] = useState<HealthGeneration | null>(null);
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [workspaceId, setWorkspaceId] = useState("");
  const [documents, setDocuments] = useState<DocItem[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [docError, setDocError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [chatError, setChatError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<{ title: string; body: string; error: string | null } | null>(
    null,
  );
  const runRef = useRef<{
    id: string | null;
    kind: "answers" | "coverage" | "calculate";
    abort: AbortController;
  } | null>(null);
  const generationRef = useRef(0);
  const polls = useRef<number[]>([]);

  useEffect(() => {
    let gone = false;
    void (async () => {
      try {
        const readyResponse = await fetch("/health/ready");
        const generationResponse = await fetch("/health/generation");
        if (gone) return;
        if (readyResponse.ok) setReady((await readyResponse.json()) as HealthReady);
        setGeneration((await generationResponse.json()) as HealthGeneration);
      } catch {
        if (!gone) {
          setReady(null);
          setGeneration(null);
        }
      }
    })();
    return () => {
      gone = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      for (const timer of polls.current) window.clearInterval(timer);
    };
  }, []);

  async function loadDocuments(currentToken: string, currentWorkspace: string, generationId: number) {
    const client = createClient(currentToken);
    const rows = await client.documents(currentWorkspace);
    if (generationRef.current !== generationId) return;
    setDocuments(rows.map(toDoc));
    await Promise.all(
      rows.map(async (row) => {
        try {
          const extraction = await client.extraction(currentWorkspace, row.version_id);
          if (generationRef.current !== generationId) return;
          setDocuments((current) =>
            current.map((doc) =>
              doc.versionId === row.version_id ? withExtraction(doc, extraction) : doc,
            ),
          );
        } catch {
          /* list status remains visible */
        }
      }),
    );
  }

  useEffect(() => {
    if (!token) return;
    const generationId = ++generationRef.current;
    const client = createClient(token);
    void client
      .workspaces()
      .then((rows) => {
        if (generationRef.current !== generationId) return;
        setWorkspaces(rows);
        setWorkspaceId((current) => current || rows[0]?.id || "");
        setAuthError(null);
      })
      .catch((error: unknown) => {
        if (generationRef.current !== generationId) return;
        if (error instanceof ApiError && error.status === 401) signOut();
        else setDocError(t("loadFailed"));
      });
  }, [token]);

  useEffect(() => {
    if (!token || !workspaceId) return;
    const generationId = ++generationRef.current;
    setSelected([]);
    setDocError(null);
    void loadDocuments(token, workspaceId, generationId).catch(() => {
      if (generationRef.current === generationId) setDocError(t("loadFailed"));
    });
  }, [token, workspaceId]);

  function signOut() {
    try {
      sessionStorage.removeItem(TOKEN_KEY);
    } catch {
      /* ignore */
    }
    runRef.current?.abort.abort();
    setToken(null);
    setWorkspaces([]);
    setWorkspaceId("");
    setDocuments([]);
    setSelected([]);
    setEntries([]);
  }

  async function signIn(value: string) {
    const next = value.trim();
    if (!next) return;
    setAuthBusy(true);
    setAuthError(null);
    try {
      const rows = await createClient(next).workspaces();
      try {
        sessionStorage.setItem(TOKEN_KEY, next);
      } catch {
        /* session storage can be unavailable; the tab state still holds the token */
      }
      setWorkspaces(rows);
      setWorkspaceId(rows[0]?.id ?? "");
      setToken(next);
    } catch (error) {
      setAuthError(
        error instanceof ApiError && error.status === 401
          ? t("authFailed")
          : error instanceof ApiError && error.message
            ? error.message
            : t("apiUnavailable"),
      );
    } finally {
      setAuthBusy(false);
    }
  }

  function watchExtraction(versionId: string) {
    if (!token || !workspaceId) return;
    const client = createClient(token);
    const currentWorkspace = workspaceId;
    const timer = window.setInterval(() => {
      void client
        .extraction(currentWorkspace, versionId)
        .then((extraction) => {
          setDocuments((current) =>
            current.map((doc) => (doc.versionId === versionId ? withExtraction(doc, extraction) : doc)),
          );
          if (extraction.status !== "queued" && extraction.status !== "extracting") {
            window.clearInterval(timer);
          }
        })
        .catch(() => window.clearInterval(timer));
    }, 2000);
    polls.current.push(timer);
  }

  async function upload(file: File) {
    if (!token || !workspaceId) return;
    if (uploadProblem(file)) {
      setDocError(t("fileRejected"));
      return;
    }
    setUploading(true);
    setDocError(null);
    try {
      const client = createClient(token);
      const created = await client.upload(workspaceId, file);
      const generationId = ++generationRef.current;
      await loadDocuments(token, workspaceId, generationId);
      watchExtraction(created.version_id);
    } catch (error) {
      setDocError(error instanceof ApiError && error.message ? error.message : t("apiUnavailable"));
    } finally {
      setUploading(false);
    }
  }

  function toggle(versionId: string) {
    setSelected((current) => {
      if (current.includes(versionId)) return current.filter((id) => id !== versionId);
      if (current.length >= 8) {
        setChatError(t("tooManyVersions"));
        return current;
      }
      setChatError(null);
      return [...current, versionId];
    });
  }

  function patchEntry(id: string, update: (entry: ChatEntry) => ChatEntry) {
    setEntries((current) => current.map((entry) => (entry.id === id ? update(entry) : entry)));
  }

  function readyCsv() {
    const chosen = documents.filter((doc) => selected.includes(doc.versionId) && doc.retrievalReady);
    const csv = chosen.filter((doc) => isCsvFilename(doc.filename));
    return { chosen, csv };
  }

  function calculationEntry(payload: CalculationResponse, downloads = readDownloads(payload.downloads)) {
    const record = readCalculationRecord(payload.record, payload.status);
    const answer: AnswerPayload = {
      event: "answer",
      mode: "spreadsheet_calculation",
      answer: typeof payload.answer === "string" ? payload.answer : record?.answer,
      limitation: record?.limitation ?? undefined,
      metadata: {
        production_qualified: payload.production_qualified === true,
        model_called: payload.model_called === true,
        calculation_id: payload.calculation_id,
        calculation_status: payload.status,
      },
    };
    return { record, downloads, events: [answer] };
  }

  async function ask(question: string, kind: "answer" | "coverage" | "calculate") {
    const text = question.trim();
    if (!token || !workspaceId) return;
    if (!text) {
      setChatError(t("questionRequired"));
      return;
    }
    const requested =
      kind === "calculate" ||
      (kind === "answer" && (text === "/calculate" || text.startsWith("/calculate ")));
    const ready = readyCsv();
    if (requested) {
      if (ready.chosen.length !== 1 || ready.csv.length !== 1) {
        setChatError(ready.chosen.length === 1 ? t("csvOnly") : t("oneCsv"));
        return;
      }
    } else if (selected.length < 1 || selected.length > 8) {
      setChatError(t("selectHint"));
      return;
    }
    const direct = kind === "calculate";
    const questionText = direct ? stripCalculate(text) : text;
    if (direct && !questionText) {
      setChatError(t("questionRequired"));
      return;
    }
    setChatError(null);
    setBusy(true);
    const assistantId = crypto.randomUUID();
    const abort = new AbortController();
    const streamKind = kind === "coverage" ? "coverage" : "answers";
    const entryKind = requested ? "calculate" : kind;
    runRef.current = { id: null, kind: requested ? "calculate" : streamKind, abort };
    setEntries((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", kind: entryKind, text: questionText },
      { id: assistantId, role: "assistant", kind: entryKind, events: [], running: true, error: null },
    ]);
    const client = createClient(token);
    if (direct) {
      const versionId = ready.csv[0]?.versionId;
      if (!versionId) {
        patchEntry(assistantId, (entry) => ({ ...entry, running: false, error: t("oneCsv") }));
        setBusy(false);
        runRef.current = null;
        return;
      }
      try {
        const payload = await client.createCalculation(
          workspaceId,
          {
            version_id: versionId,
            question: questionText,
            client_request_id: crypto.randomUUID(),
          },
          abort.signal,
        );
        const attached = calculationEntry(payload);
        patchEntry(assistantId, (entry) => ({
          ...entry,
          running: false,
          events: attached.events,
          calculation: attached.record,
          downloads: attached.downloads,
          error: attached.record ? null : t("apiUnavailable"),
        }));
      } catch (error) {
        patchEntry(assistantId, (entry) => ({
          ...entry,
          running: false,
          error: abort.signal.aborted
            ? t("cancelled")
            : error instanceof ApiError && error.message
              ? error.message
              : t("apiUnavailable"),
        }));
      } finally {
        setBusy(false);
        runRef.current = null;
      }
      return;
    }
    const events: StreamEvent[] = [];
    try {
      const response = await client.openStream(
        workspaceId,
        streamKind,
        {
          question: text,
          version_ids: requested && ready.csv[0] ? [ready.csv[0].versionId] : selected,
          client_request_id: crypto.randomUUID(),
        },
        abort.signal,
      );
      await readNdjson(response, (event) => {
        events.push(event);
        if (event.event === "queued" && event.run_id && runRef.current) {
          runRef.current.id = event.run_id;
        }
        patchEntry(assistantId, (entry) => ({ ...entry, events: [...events] }));
      });
      const failure = [...events].reverse().find((event) => event.event === "error");
      if (failure && failure.event === "error" && failure.code === "stream_limit" && kind === "coverage") {
        const runId = runRef.current?.id;
        if (runId) {
          const map = await client.coverageMap(workspaceId, runId);
          const coverage = { event: "coverage" as const, ...map };
          events.push(coverage);
          patchEntry(assistantId, (entry) => ({
            ...entry,
            events: [...events],
            error: t("streamInterrupted"),
          }));
        }
      } else if (failure && failure.event === "error") {
        patchEntry(assistantId, (entry) => ({
          ...entry,
          error: failure.detail || t("streamInterrupted"),
        }));
      } else if (!events.some((event) => event.event === "answer" || event.event === "error")) {
        patchEntry(assistantId, (entry) => ({ ...entry, error: t("streamInterrupted") }));
      }
      const released = [...events].reverse().find((event) => event.event === "answer");
      if (released && released.event === "answer" && released.mode === "spreadsheet_calculation") {
        const id = calculationId(released.metadata);
        if (id) {
          const body = await client.calculation(workspaceId, id);
          const supplied = readDownloads(released.downloads);
          const downloads = {
            record:
              supplied.record ??
              calculationDownloadPath(
                `/api/workspaces/${workspaceId}/calculations/${id}/record.json`,
              ),
            table:
              supplied.table ??
              (body.status === "completed"
                ? calculationDownloadPath(`/api/workspaces/${workspaceId}/calculations/${id}/table.csv`)
                : null),
          };
          patchEntry(assistantId, (entry) => ({
            ...entry,
            calculation: readCalculationRecord(body.record, body.status),
            downloads,
          }));
        }
      }
    } catch (error) {
      if (abort.signal.aborted) {
        patchEntry(assistantId, (entry) => ({ ...entry, error: entry.error ?? t("cancelled") }));
      } else {
        patchEntry(assistantId, (entry) => ({
          ...entry,
          error: error instanceof ApiError && error.message ? error.message : t("apiUnavailable"),
        }));
      }
    } finally {
      patchEntry(assistantId, (entry) => ({ ...entry, running: false }));
      setBusy(false);
      runRef.current = null;
    }
  }

  async function cancel() {
    const active = runRef.current;
    active?.abort.abort();
    if (token && workspaceId && active?.id && active.kind !== "calculate") {
      try {
        await createClient(token).cancelRun(workspaceId, active.kind, active.id);
      } catch {
        /* the aborted stream already stops the local view */
      }
    }
  }

  async function download(path: string) {
    const safe = calculationDownloadPath(path);
    if (!safe || !token) {
      setChatError(t("downloadRejected"));
      return;
    }
    try {
      const response = await createClient(token).download(safe);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = safe.endsWith("/table.csv") ? "calculation.csv" : "calculation-record.json";
      anchor.target = "_blank";
      anchor.rel = "noopener";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (error) {
      setChatError(error instanceof ApiError && error.message ? error.message : t("downloadRejected"));
    }
  }

  async function openPreview(path: string, title: string) {
    const safe = previewPath(path);
    if (!safe || !token) {
      setPreview({ title, body: "", error: t("previewUnavailable") });
      return;
    }
    setPreview({ title, body: "", error: null });
    try {
      const body = await createClient(token).preview(safe);
      setPreview({ title, body, error: null });
    } catch (error) {
      setPreview({
        title,
        body: "",
        error: error instanceof ApiError && error.message ? error.message : t("previewUnavailable"),
      });
    }
  }

  const indexedSelection = selected.filter((id) =>
    documents.some((doc) => doc.versionId === id && doc.retrievalReady),
  );

  return (
    <div className="page">
      <header className="top">
        <div>
          <p className="brand">{t("product")}</p>
          <p className="note">{t("developmentOnly")}</p>
          <HealthStrip ready={ready} generation={generation} />
        </div>
        <div className="top-actions">
          <LanguageToggle />
          {token ? (
            <button type="button" className="ghost" onClick={signOut}>
              {t("signOut")}
            </button>
          ) : null}
        </div>
      </header>
      <main id="main">
        {!token ? <SignIn busy={authBusy} error={authError} onSubmit={(value) => void signIn(value)} /> : null}
        {token ? (
          <div className="workspace">
            <DocumentPanel
              workspaces={workspaces}
              workspaceId={workspaceId}
              documents={documents}
              selected={selected}
              uploading={uploading}
              error={docError}
              onWorkspace={setWorkspaceId}
              onToggle={toggle}
              onUpload={(file) => void upload(file)}
              onCancelExtraction={(versionId) => {
                if (!token || !workspaceId) return;
                void createClient(token)
                  .cancelExtraction(workspaceId, versionId)
                  .then(() => watchExtraction(versionId))
                  .catch((error: unknown) => {
                    setDocError(error instanceof ApiError && error.message ? error.message : t("apiUnavailable"));
                  });
              }}
              onRefresh={() => {
                const generationId = ++generationRef.current;
                void loadDocuments(token, workspaceId, generationId).catch(() => setDocError(t("loadFailed")));
              }}
            />
            <ChatPanel
              entries={entries}
              busy={busy}
              error={chatError}
              canAsk={indexedSelection.length > 0}
              canCalculate={
                indexedSelection.length === 1 &&
                documents.some(
                  (doc) => doc.versionId === indexedSelection[0] && isCsvFilename(doc.filename),
                )
              }
              onAsk={(question, kind) => void ask(question, kind)}
              onCancel={() => void cancel()}
              onPreview={(path, title) => void openPreview(path, title)}
              onDownload={(path) => void download(path)}
            />
          </div>
        ) : null}
      </main>
      {preview ? (
        <SourcePreview
          title={preview.title}
          body={preview.body}
          error={preview.error}
          onClose={() => setPreview(null)}
        />
      ) : null}
    </div>
  );
}

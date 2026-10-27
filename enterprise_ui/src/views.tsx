import type { FormEvent, ReactNode } from "react";
import { CalculationView } from "./calculations/CalculationView";
import type { CalculationDownloads, CalculationRecordView } from "./calculations/record";
import {
  coverageIsComplete,
  locationParts,
  previewPath,
  readDisclosure,
  sectionKey,
} from "./disclosure";
import { formatMessage, statusLabel, useLocale, type MessageKey } from "./i18n";
import type { AnswerPayload, Claim, CoveragePayload, HealthGeneration, HealthReady, SectionRow, StreamEvent } from "./types";
import { BidiText, Isolate } from "./bidi";

export type DocItem = {
  id: string;
  filename: string;
  versionId: string;
  versionNumber: number;
  byteSize: number;
  sha256: string;
  ingestionStatus: string;
  retrievalReady: boolean | null;
  warnings: string[];
  errorDetail: string | null;
};

export type ChatEntry = {
  id: string;
  role: "user" | "assistant";
  kind: "answer" | "coverage" | "calculate";
  text?: string;
  events?: StreamEvent[];
  error?: string | null;
  running?: boolean;
  calculation?: CalculationRecordView | null;
  downloads?: CalculationDownloads | null;
};

function locationLabel(location: unknown, t: (key: MessageKey) => string): string {
  const parts = locationParts(location);
  if (parts.kind === "page") return `${t("page")} ${parts.page}`;
  if (parts.kind === "row") return `${t("rowLabel")} ${parts.row}`;
  if (parts.kind === "lines") return `${t("lines")} ${parts.start}–${parts.end}`;
  return "";
}

export function LanguageToggle() {
  const { locale, toggle } = useLocale();
  const next = locale === "ar" ? "English" : "العربية";
  return (
    <button
      type="button"
      className="ghost"
      onClick={toggle}
      lang={locale === "ar" ? "en" : "ar"}
      dir={locale === "ar" ? "ltr" : "rtl"}
      aria-pressed={locale === "ar"}
    >
      {next}
    </button>
  );
}

export function SignIn({
  busy,
  error,
  onSubmit,
}: {
  busy: boolean;
  error: string | null;
  onSubmit: (token: string) => void;
}) {
  const { t } = useLocale();
  return (
    <form
      className="sign-in"
      onSubmit={(event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        const data = new FormData(event.currentTarget);
        onSubmit(String(data.get("token") ?? ""));
      }}
    >
      <h1>{t("signInTitle")}</h1>
      <p>{t("signInHelp")}</p>
      <p className="note">{t("identityNote")}</p>
      <label>
        {t("tokenLabel")}
        <input name="token" type="password" autoComplete="off" spellCheck={false} required />
      </label>
      {error ? <p role="alert">{error}</p> : null}
      <button type="submit" disabled={busy}>
        {busy ? t("signingIn") : t("signIn")}
      </button>
    </form>
  );
}

export function HealthStrip({
  ready,
  generation,
}: {
  ready: HealthReady | null;
  generation: HealthGeneration | null;
}) {
  const { t } = useLocale();
  return (
    <p className="health">
      <span>{t("generation")}: </span>
      <Isolate>{generation?.mode ?? ready?.generation ?? t("notReported")}</Isolate>
      <span className="dot"> · </span>
      <span>{generation?.production_qualified ? t("productionQualified") : t("productionUnqualified")}</span>
    </p>
  );
}

export function DocumentPanel({
  workspaces,
  workspaceId,
  documents,
  selected,
  uploading,
  error,
  onWorkspace,
  onToggle,
  onUpload,
  onCancelExtraction,
  onRefresh,
}: {
  workspaces: { id: string; name: string }[];
  workspaceId: string;
  documents: DocItem[];
  selected: string[];
  uploading: boolean;
  error: string | null;
  onWorkspace: (id: string) => void;
  onToggle: (versionId: string) => void;
  onUpload: (file: File) => void;
  onCancelExtraction: (versionId: string) => void;
  onRefresh: () => void;
}) {
  const { t } = useLocale();
  return (
    <section className="panel" aria-label={t("documents")}>
      <header className="panel-head">
        <h2>{t("workspaces")}</h2>
        <button type="button" className="ghost" onClick={onRefresh}>
          {t("refresh")}
        </button>
      </header>
      {workspaces.length === 0 ? <p>{t("noWorkspaces")}</p> : null}
      <div className="workspace-list">
        {workspaces.map((workspace) => (
          <button
            key={workspace.id}
            type="button"
            className={workspace.id === workspaceId ? "choice active" : "choice"}
            aria-pressed={workspace.id === workspaceId}
            onClick={() => onWorkspace(workspace.id)}
          >
            <BidiText text={workspace.name} />
          </button>
        ))}
      </div>
      <h2>{t("documents")}</h2>
      <p className="note">{t("selectHint")}</p>
      <p className="note">{formatMessage(t("selectedCount"), { count: selected.length })}</p>
      {documents.length === 0 ? <p>{t("noDocuments")}</p> : null}
      <ul className="documents">
        {documents.map((doc) => {
          const active = doc.ingestionStatus === "queued" || doc.ingestionStatus === "extracting";
          return (
            <li key={doc.versionId}>
              <label>
                <input
                  type="checkbox"
                  checked={selected.includes(doc.versionId)}
                  disabled={doc.retrievalReady !== true}
                  onChange={() => onToggle(doc.versionId)}
                />
                <span>
                  <BidiText text={doc.filename} />
                  <small>
                    <Isolate>v{doc.versionNumber}</Isolate>
                    {" · "}
                    {statusLabel(doc.ingestionStatus, t)}
                    {" · "}
                    <Isolate>
                      {doc.byteSize} {t("bytes")}
                    </Isolate>
                  </small>
                  <small>
                    {doc.retrievalReady ? t("ready") : t("notIndexed")}
                    {doc.sha256 ? (
                      <>
                        {" · "}
                        <Isolate>{doc.sha256.slice(0, 12)}</Isolate>
                      </>
                    ) : null}
                  </small>
                </span>
              </label>
              {doc.warnings.length > 0 ? (
                <p className="warn">
                  {t("warnings")}: <BidiText text={doc.warnings.join(" · ")} />
                </p>
              ) : null}
              {doc.errorDetail ? (
                <p className="warn" role="status">
                  <BidiText text={doc.errorDetail} />
                </p>
              ) : null}
              {active ? (
                <button type="button" className="ghost" onClick={() => onCancelExtraction(doc.versionId)}>
                  {t("cancelExtraction")}
                </button>
              ) : null}
            </li>
          );
        })}
      </ul>
      <label className="upload">
        {uploading ? t("uploadBusy") : t("upload")}
        <input
          type="file"
          accept=".pdf,.txt,.md,.csv,application/pdf,text/plain,text/markdown,text/csv"
          disabled={uploading || !workspaceId}
          onChange={(event) => {
            const file = event.currentTarget.files?.[0];
            event.currentTarget.value = "";
            if (file) onUpload(file);
          }}
        />
      </label>
      <p className="note">{t("uploadHint")}</p>
      {error ? <p role="alert">{error}</p> : null}
    </section>
  );
}

function stageCopy(events: StreamEvent[], t: (key: MessageKey) => string): string | null {
  const section = [...events].reverse().find((event) => event.event === "section");
  if (section && section.event === "section" && section.total !== undefined) {
    return formatMessage(t("sectionProgress"), {
      covered: section.covered ?? 0,
      total: section.total,
    });
  }
  const stage = [...events].reverse().find((event) => event.event === "stage" || event.event === "queued");
  if (!stage) return null;
  if (stage.event === "queued") return t("stageQueued");
  if (stage.stage === "generating_and_validating") return t("stageGenerating");
  if (stage.stage === "covering_sections") return t("stageCovering");
  if (stage.stage === "reducing_sections") return t("stageReducing");
  if (stage.stage === "calculating") return t("stageCalculating");
  return stage.stage ?? t("stageQueued");
}

function mapPayload(events: StreamEvent[]): AnswerPayload | CoveragePayload | null {
  const answer = [...events].reverse().find((event) => event.event === "answer");
  if (answer && answer.event === "answer" && answer.mode === "exhaustive_coverage") return answer;
  const coverage = [...events].reverse().find((event) => event.event === "coverage");
  return coverage && coverage.event === "coverage" ? coverage : null;
}

export function CoverageMap({ payload }: { payload: AnswerPayload | CoveragePayload }) {
  const { t } = useLocale();
  const sections = payload.sections ?? [];
  const complete = coverageIsComplete(payload);
  return (
    <section className="coverage" aria-label={t("coverageMap")}>
      <p className={complete ? "badge ok" : "badge warn"}>
        {complete ? t("complete") : t("partial")}
      </p>
      {payload.scope_notice ? (
        <p>
          <strong>{t("scope")}: </strong>
          <BidiText text={payload.scope_notice} />
        </p>
      ) : null}
      <table>
        <thead>
          <tr>
            <th scope="col">{t("filename")}</th>
            <th scope="col">{t("location")}</th>
            <th scope="col">{t("status")}</th>
            <th scope="col">{t("gapCode")}</th>
          </tr>
        </thead>
        <tbody>
          {sections.map((section: SectionRow, index) => (
            <tr key={sectionKey(section, index)}>
              <td>
                <BidiText text={section.filename ?? ""} />
              </td>
              <td>{locationLabel(section.location, t)}</td>
              <td>{section.status === "covered" ? t("covered") : t("gaps")}</td>
              <td>
                <Isolate>{section.gap_code ?? ""}</Isolate>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function AnswerCard({
  entry,
  onPreview,
  onDownload,
}: {
  entry: ChatEntry;
  onPreview: (path: string, title: string) => void;
  onDownload?: (path: string) => void;
}) {
  const { t } = useLocale();
  const events = entry.events ?? [];
  const answer = [...events].reverse().find((event) => event.event === "answer");
  const payload = answer && answer.event === "answer" ? answer : null;
  const disclosure = payload ? readDisclosure(payload) : null;
  const queued = events.find((event) => event.event === "queued");
  const external = queued && queued.event === "queued" && queued.external_processing === true;
  const coverage = mapPayload(events);
  return (
    <article className="bubble assistant">
      {entry.running ? <p className="stage">{stageCopy(events, t) ?? t("stillProcessing")}</p> : null}
      {external ? <p className="note">{t("externalProcessing")}</p> : null}
      {queued && queued.event === "queued" && queued.privacy_policy === "nonconfidential-development" ? (
        <p className="note">{t("privacyStandard")}</p>
      ) : null}
      {payload?.answer ? <BidiText as="div" className="prose" text={payload.answer} /> : null}
      {payload?.insufficient && !payload.answer ? <p>{t("noModelCall")}</p> : null}
      {disclosure ? (
        <dl className="disclosure">
          <div>
            <dt>{t("model")}</dt>
            <dd>
              <Isolate>{disclosure.model ?? t("notReported")}</Isolate>
            </dd>
          </div>
          <div>
            <dt>{t("provider")}</dt>
            <dd>
              <Isolate>{disclosure.provider ?? t("notReported")}</Isolate>
            </dd>
          </div>
          {disclosure.requestedModel ? (
            <div>
              <dt>{t("requestedModel")}</dt>
              <dd>
                <Isolate>{disclosure.requestedModel}</Isolate>
              </dd>
            </div>
          ) : null}
          {payload?.mode === "spreadsheet_calculation" ? null : (
          <div>
            <dt>{t("contextCoverage")}</dt>
            <dd>
              {disclosure.scopeNotice ? <BidiText text={disclosure.scopeNotice} /> : null}
              {disclosure.total !== null ? (
                <span>
                  {" "}
                  {formatMessage(t("passages"), {
                    included: disclosure.included ?? 0,
                    total: disclosure.total,
                  })}
                </span>
              ) : null}
              {disclosure.allIncluded === false ? <span> {t("partialOverview")}</span> : null}
            </dd>
          </div>
          )}
          {payload?.mode === "spreadsheet_calculation" ? (
            <div>
              <dt>{t("model")}</dt>
              <dd>{disclosure.modelCalled ? t("plannerCalled") : t("explicitPlan")}</dd>
            </div>
          ) : disclosure.modelCalled === false ? (
            <div>
              <dt>{t("model")}</dt>
              <dd>{t("noModelCall")}</dd>
            </div>
          ) : null}
          {disclosure.fixture ? (
            <div>
              <dt>{t("generation")}</dt>
              <dd>{t("fixtureNotice")}</dd>
            </div>
          ) : null}
          <div>
            <dt>{t("developmentOnly")}</dt>
            <dd>
              {disclosure.productionQualified ? t("productionQualified") : t("productionUnqualified")}
            </dd>
          </div>
        </dl>
      ) : null}
      {payload?.limitation ? (
        <p className="note">
          <strong>{t("limitation")}: </strong>
          <BidiText text={payload.limitation} />
        </p>
      ) : null}
      {payload?.claims && payload.claims.length > 0 ? (
        <div>
          <h3>{t("sources")}</h3>
          <ul className="sources">
            {payload.claims.map((claim: Claim, index) => {
              const path = previewPath(claim.preview_url);
              const title = claim.filename ?? claim.source_id ?? String(index + 1);
              return (
                <li key={`${claim.source_id ?? "source"}-${index}`}>
                  <p>
                    <BidiText text={title} />
                    {claim.location ? <span> · {locationLabel(claim.location, t)}</span> : null}
                  </p>
                  {claim.quote ? <blockquote><BidiText text={claim.quote} /></blockquote> : null}
                  {path ? (
                    <button type="button" className="ghost" onClick={() => onPreview(path, title)}>
                      {t("preview")}
                    </button>
                  ) : null}
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
      {coverage ? <CoverageMap payload={coverage} /> : null}
      {entry.calculation ? (
        <CalculationView
          record={entry.calculation}
          downloads={entry.downloads}
          onDownload={onDownload ?? (() => undefined)}
        />
      ) : (
        <p className="note">{t("notCalculation")}</p>
      )}
      {entry.error ? <p role="alert">{entry.error}</p> : null}
    </article>
  );
}

export function ChatPanel({
  entries,
  busy,
  error,
  canAsk,
  canCalculate,
  onAsk,
  onCancel,
  onPreview,
  onDownload,
}: {
  entries: ChatEntry[];
  busy: boolean;
  error: string | null;
  canAsk: boolean;
  canCalculate: boolean;
  onAsk: (question: string, kind: "answer" | "coverage" | "calculate") => void;
  onCancel: () => void;
  onPreview: (path: string, title: string) => void;
  onDownload: (path: string) => void;
}) {
  const { t } = useLocale();
  return (
    <section className="chat" aria-label={t("questionLabel")}>
      <p className="note">{t("chatHint")}</p>
      <p className="note">{t("calculationHint")}</p>
      <div className="transcript" role="log" aria-live="polite">
        {entries.length === 0 ? <p className="empty">{t("emptyChat")}</p> : null}
        {entries.map((entry) =>
          entry.role === "user" ? (
            <article key={entry.id} className="bubble user">
              <p className="kind">
                {entry.kind === "coverage" ? t("coverage") : entry.kind === "calculate" ? t("calculate") : t("ask")}
              </p>
              <BidiText as="div" className="prose" text={entry.text ?? ""} />
            </article>
          ) : (
            <AnswerCard key={entry.id} entry={entry} onPreview={onPreview} onDownload={onDownload} />
          ),
        )}
      </div>
      <form
        className="composer"
        onSubmit={(event: FormEvent<HTMLFormElement>) => {
          event.preventDefault();
          const data = new FormData(event.currentTarget);
          onAsk(String(data.get("question") ?? ""), "answer");
        }}
      >
        <label>
          {t("questionLabel")}
          <textarea
            name="question"
            maxLength={512}
            rows={3}
            placeholder={t("questionPlaceholder")}
            required
          />
        </label>
        {error ? <p role="alert">{error}</p> : null}
        <div className="actions">
          <button type="submit" disabled={!canAsk || busy}>
            {t("ask")}
          </button>
          <button
            type="button"
            className="secondary"
            disabled={!canAsk || busy}
            onClick={(event) => {
              const form = event.currentTarget.form;
              const question = String(new FormData(form ?? undefined).get("question") ?? "");
              onAsk(question, "coverage");
            }}
          >
            {t("coverage")}
          </button>
          <button
            type="button"
            className="secondary"
            disabled={!canCalculate || busy}
            onClick={(event) => {
              const form = event.currentTarget.form;
              const question = String(new FormData(form ?? undefined).get("question") ?? "");
              onAsk(question, "calculate");
            }}
          >
            {t("calculate")}
          </button>
          {busy ? (
            <button type="button" className="ghost" onClick={onCancel}>
              {t("cancel")}
            </button>
          ) : null}
        </div>
      </form>
    </section>
  );
}

export function SourcePreview({
  title,
  body,
  error,
  onClose,
}: {
  title: string;
  body: string;
  error: string | null;
  onClose: () => void;
}) {
  const { t } = useLocale();
  return (
    <div className="preview-backdrop" role="presentation" onClick={onClose}>
      <div
        className="preview"
        role="dialog"
        aria-modal="true"
        aria-label={t("previewTitle")}
        onClick={(event) => event.stopPropagation()}
      >
        <header>
          <h2>{t("previewTitle")}</h2>
          <button type="button" className="ghost" onClick={onClose}>
            {t("close")}
          </button>
        </header>
        <p>
          <BidiText text={title} />
        </p>
        <p className="note">{t("nativePreviewNote")}</p>
        {error ? <p role="alert">{error}</p> : <pre>{body}</pre>}
      </div>
    </div>
  );
}

export function Frame({ children }: { children: ReactNode }) {
  const { t } = useLocale();
  return (
    <>
      <a className="skip" href="#main">
        {t("questionLabel")}
      </a>
      {children}
    </>
  );
}

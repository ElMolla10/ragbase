import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { readNdjson, uploadProblem } from "../api";
import { BidiText } from "../bidi";
import { CalculationView } from "../calculations/CalculationView";
import {
  calculationDownloadPath,
  readCalculationRecord,
  type CalculationRecordView,
} from "../calculations/record";
import { coverageIsComplete, previewPath, readDisclosure } from "../disclosure";
import { LocaleProvider } from "../i18n";
import type { AnswerPayload } from "../types";
import { AnswerCard, CoverageMap, type ChatEntry } from "../views";

function renderUi(node: ReactNode) {
  return render(<LocaleProvider>{node}</LocaleProvider>);
}

const workspace = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
const version = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("direction", () => {
  it("switches the document to Arabic RTL and isolates mixed text", async () => {
    const user = userEvent.setup();
    renderUi(<App />);
    expect(document.documentElement).toHaveAttribute("dir", "ltr");
    await user.click(screen.getByRole("button", { name: "العربية" }));
    await waitFor(() => expect(document.documentElement).toHaveAttribute("dir", "rtl"));
    expect(document.documentElement).toHaveAttribute("lang", "ar");
    expect(screen.getByRole("heading", { name: "الدخول بهوية تطوير" })).toBeInTheDocument();

    render(<BidiText text={"Quarter 1 المبيعات 120"} />);
    const mixed = screen.getByText("Quarter 1 المبيعات 120");
    expect(mixed).toHaveAttribute("dir", "auto");
    expect(mixed).toHaveStyle({ unicodeBidi: "isolate" });
  });
});

describe("sign in", () => {
  it("sends the development token and does not call workspaces before submit", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/health/ready") return json({ generation: "disabled", status: "storage_ready" });
      if (url === "/health/generation") {
        return json({ mode: "disabled", production_qualified: false }, 503);
      }
      if (url === "/api/workspaces") {
        expect(new Headers(init?.headers).get("authorization")).toBe("Bearer dev-token-value");
        return json({ workspaces: [{ id: workspace, name: "Synthetic workspace A" }] });
      }
      if (url.endsWith("/documents")) {
        return json({
          documents: [
            {
              id: "doc",
              filename: "تقرير Q1.pdf",
              version_id: version,
              version_number: 1,
              byte_size: 120,
              sha256: "abc123abc123",
              status: "stored",
              ingestion_status: "extracted",
            },
          ],
        });
      }
      if (url.endsWith("/extraction")) {
        return json({ status: "extracted", retrieval_ready: true, warnings: ["blank page"] });
      }
      return json({ detail: "missing" }, 404);
    });
    vi.stubGlobal("fetch", fetchMock);
    renderUi(<App />);
    expect(fetchMock.mock.calls.some((call) => String(call[0]) === "/api/workspaces")).toBe(false);
    await user.type(screen.getByLabelText("Development token"), "dev-token-value");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByText("Synthetic workspace A")).toBeInTheDocument();
    expect(await screen.findByText("تقرير Q1.pdf")).toBeInTheDocument();
    expect(screen.getByText("تقرير Q1.pdf")).toHaveAttribute("dir", "auto");
  });

  it("keeps the sign-in form when the token is rejected", async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.startsWith("/health/")) return json({ mode: "disabled" }, 200);
        return json({ detail: "Invalid development token" }, 401);
      }),
    );
    renderUi(<App />);
    await user.type(screen.getByLabelText("Development token"), "not-a-real-token");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The development token was not accepted.");
    expect(screen.queryByText("Synthetic workspace A")).not.toBeInTheDocument();
  });
});

describe("answers and coverage", () => {
  const answer: AnswerPayload = {
    event: "answer",
    mode: "grounded_document_chat",
    answer: "Quarter 1 sales were 120 (Page 1).",
    claims: [
      {
        source_id: "s1",
        quote: "Quarter 1 sales: 120",
        filename: "sales.pdf",
        location: { page: 1 },
        preview_url: `/api/workspaces/${workspace}/sources/${version}`,
      },
    ],
    scope_notice: "Document context: 1 of 4 extracted passages supplied.",
    limitation: "Generated explanations are model interpretations.",
    metadata: {
      fixture: false,
      model_called: true,
      external_processing: true,
      model: {
        requested_model: "openai/gpt-oss-120b",
        production_qualified: false,
        resolved_route: { model: "openai/gpt-oss-120b", provider: "groq" },
      },
      coverage: {
        included_chunks: 1,
        total_document_chunks: 4,
        all_extracted_chunks_included: false,
      },
    },
  };

  it("shows the reported model, provider, partial context, and plain-text preview", async () => {
    const user = userEvent.setup();
    const entry: ChatEntry = {
      id: "1",
      role: "assistant",
      kind: "answer",
      events: [{ event: "queued", external_processing: true }, answer],
    };
    const onPreview = vi.fn();
    renderUi(<AnswerCard entry={entry} onPreview={onPreview} />);
    expect(screen.getAllByText("openai/gpt-oss-120b").length).toBeGreaterThan(0);
    expect(screen.getByText("groq")).toBeInTheDocument();
    expect(screen.getByText(/1 of 4 passages supplied/)).toBeInTheDocument();
    expect(screen.getByText(/Partial overview/)).toBeInTheDocument();
    expect(screen.getByText("Not production qualified")).toBeInTheDocument();
    expect(screen.getByText("Quarter 1 sales were 120 (Page 1).")).toHaveAttribute("dir", "auto");
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Preview source" }));
    expect(onPreview).toHaveBeenCalledWith(
      `/api/workspaces/${workspace}/sources/${version}`,
      "sales.pdf",
    );
    const disclosure = readDisclosure(answer);
    expect(disclosure.model).toBe("openai/gpt-oss-120b");
    expect(disclosure.provider).toBe("groq");
    expect(disclosure.allIncluded).toBe(false);
  });

  it("renders a covered and gap section map without treating a partial run as complete", () => {
    renderUi(
      <CoverageMap
        payload={{
          event: "coverage",
          label: "complete",
          coverage_complete: false,
          scope_notice: "Partial coverage: 1 of 2 sections processed.",
          sections: [
            {
              filename: "ملاحظات.md",
              status: "covered",
              location: { line_start: 1, line_end: 2 },
              ordinal: 1,
              part: 0,
            },
            {
              filename: "ملاحظات.md",
              status: "gap",
              gap_code: "blank_or_unreadable",
              location: { page: 2 },
              ordinal: 2,
              part: 0,
            },
          ],
        }}
      />,
    );
    expect(screen.getByText("Partial")).toBeInTheDocument();
    expect(screen.queryByText("Complete")).not.toBeInTheDocument();
    expect(screen.getByText("blank_or_unreadable")).toBeInTheDocument();
    expect(screen.getAllByText("ملاحظات.md")[0]).toHaveAttribute("dir", "auto");
    expect(coverageIsComplete({ label: "complete", coverage_complete: false })).toBe(false);
    expect(coverageIsComplete({ label: "complete", coverage_complete: true })).toBe(true);
  });

  it("leaves formula text literal in a server result table", () => {
    const record = readCalculationRecord(completedRecord().record, "completed");
    expect(record?.tableRows[1]?.[1]).toBe("=1+1");
    expect(record?.tableRows[1]?.[1]).not.toBe("2");
    renderUi(
      <CalculationView
        record={record as CalculationRecordView}
        downloads={completedRecord().downloads}
        onDownload={() => undefined}
      />,
    );
    expect(screen.getByText("=1+1")).toBeInTheDocument();
    expect(screen.getAllByText(/30\.60/).length).toBeGreaterThan(0);
    expect(screen.getByText("مبيعات.csv")).toHaveAttribute("dir", "auto");
    expect(screen.getByText("sum(المبيعات)")).toHaveAttribute("dir", "auto");
    expect(screen.getByText("USD")).toBeInTheDocument();
    expect(screen.getByText("Quarter eq Q1")).toBeInTheDocument();
    expect(screen.getByText("Included rows").parentElement).toHaveTextContent("3");
    expect(screen.getByText("Excluded rows").parentElement).toHaveTextContent("1");
    expect(screen.getByRole("button", { name: "Download CSV" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download JSON" })).toBeInTheDocument();
  });

  it("shows a clarification without a result table or CSV download", async () => {
    const user = userEvent.setup();
    renderUi(<App />);
    await user.click(screen.getByRole("button", { name: "العربية" }));
    await waitFor(() => expect(document.documentElement).toHaveAttribute("dir", "rtl"));
    const record = readCalculationRecord(clarificationRecord().record, "clarification");
    renderUi(
      <CalculationView
        record={record as CalculationRecordView}
        downloads={{ table: null, record: clarificationRecord().downloads.record }}
        onDownload={() => undefined}
      />,
    );
    expect(screen.getByText("لم يُحسب أي رقم.")).toBeInTheDocument();
    expect(screen.getByText(/لم يُحسب الهامش/)).toHaveAttribute("dir", "auto");
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "تنزيل CSV" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "تنزيل JSON" })).toBeInTheDocument();
    expect(document.documentElement).toHaveAttribute("dir", "rtl");
  });
});

const calculation = "cccccccc-cccc-4ccc-8ccc-cccccccccccc";

function completedRecord() {
  return {
    downloads: {
      table: `/api/workspaces/${workspace}/calculations/${calculation}/table.csv`,
      record: `/api/workspaces/${workspace}/calculations/${calculation}/record.json`,
    },
    record: {
      answer: "Result 30.60 from sum(المبيعات).",
      dataset: {
        filename: "مبيعات.csv",
        version_id: version,
        sha256: "ab".repeat(32),
        columns: [
          { name: "المنطقة", type: "text" },
          { name: "المبيعات", type: "decimal" },
        ],
      },
      filters: [{ column: "Quarter", operator: "eq", value: "Q1" }],
      operation: {
        aggregates: [{ function: "sum", column: "المبيعات", output: "result" }],
        group_by: [],
        project: [],
        limit: null,
      },
      units: "USD",
      included_row_count: 3,
      excluded_row_count: 1,
      exclusions: { header: 1, subtotal: 0, filter: 0 },
      result: {
        columns: ["region", "result"],
        rows: [
          { region: "Q1", result: "30.60" },
          { region: "formula", result: "=1+1" },
        ],
      },
      formula_note: "Formula-like CSV text is stored literally and is not executed.",
      limitation: "Not a forecast and not production qualification.",
    },
  };
}

function clarificationRecord() {
  return {
    downloads: {
      record: `/api/workspaces/${workspace}/calculations/${calculation}/record.json`,
    },
    record: {
      status: "clarification",
      code: "ambiguous_margin",
      answer: "لم يُحسب الهامش لأن تعريفه غير محدد.",
      dataset: {
        filename: "مبيعات.csv",
        version_id: version,
        columns: [{ name: "revenue", type: "decimal" }],
      },
      filters: [],
      operation: null,
      units: "unspecified",
      included_row_count: null,
      excluded_row_count: null,
      result: null,
      limitation: "No value was calculated.",
    },
  };
}

describe("contracts", () => {
  it("parses split NDJSON and rejects unsafe preview paths", async () => {
    const events: unknown[] = [];
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('{"event":"queued","run_id":"x"}\n{"event":'));
        controller.enqueue(new TextEncoder().encode('"answer","answer":"مرحبا"}\n'));
        controller.close();
      },
    });
    await readNdjson(new Response(stream), (event) => events.push(event));
    expect(events).toEqual([
      { event: "queued", run_id: "x" },
      { event: "answer", answer: "مرحبا" },
    ]);
    expect(previewPath(`/api/workspaces/${workspace}/sources/${version}`)).toContain("/api/");
    expect(previewPath("https://example.test/secret")).toBeNull();
    expect(previewPath("/api/../.env")).toBeNull();
    const table = `/api/workspaces/${workspace}/calculations/${calculation}/table.csv`;
    expect(calculationDownloadPath(table)).toBe(table);
    expect(calculationDownloadPath(`${table}/../record.json`)).toBeNull();
    expect(calculationDownloadPath("https://example.test/table.csv")).toBeNull();
    expect(uploadProblem({ name: "notes.md", size: 10 })).toBeNull();
    expect(uploadProblem({ name: "sheet.xlsx", size: 10 })).toBe("type");
    expect(uploadProblem({ name: "big.pdf", size: 5_000_001 })).toBe("size");
  });
});

describe("calculate control", () => {
  function workspaceFetch(extra?: (url: string, init?: RequestInit) => Response | Promise<Response>) {
    return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const handled = extra ? await extra(url, init) : undefined;
      if (handled) return handled;
      if (url === "/health/ready") return json({ generation: "disabled", status: "storage_ready" });
      if (url === "/health/generation") return json({ mode: "fixture", production_qualified: false });
      if (url === "/api/workspaces") {
        return json({ workspaces: [{ id: workspace, name: "Synthetic workspace A" }] });
      }
      if (url.endsWith("/documents")) {
        return json({
          documents: [
            {
              id: "doc",
              filename: "مبيعات.csv",
              version_id: version,
              version_number: 1,
              byte_size: 40,
              sha256: "abc123abc123",
              status: "stored",
              ingestion_status: "extracted",
            },
          ],
        });
      }
      if (url.endsWith("/extraction")) {
        return json({ status: "extracted", retrieval_ready: true, warnings: [] });
      }
      return json({ detail: "missing" }, 404);
    });
  }

  it("posts one CSV question and renders the server record and download", async () => {
    const user = userEvent.setup();
    const fetchMock = workspaceFetch(async (url, init) => {
      if (url.endsWith("/calculations") && init?.method === "POST") {
        expect(new Headers(init.headers).get("authorization")).toBe("Bearer dev-token-value");
        const body = JSON.parse(String(init.body)) as { version_id: string; question: string };
        expect(body.version_id).toBe(version);
        expect(body.question).toBe("sum المبيعات");
        return json({
          calculation_id: calculation,
          status: "completed",
          answer: completedRecord().record.answer,
          record: completedRecord().record,
          downloads: completedRecord().downloads,
          model_called: true,
          production_qualified: false,
        });
      }
      if (url.endsWith("/table.csv")) {
        expect(new Headers(init?.headers).get("authorization")).toBe("Bearer dev-token-value");
        return new Response("result\n30.60\n", { status: 200, headers: { "content-type": "text/csv" } });
      }
      return undefined as unknown as Response;
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: () => "blob:calculation",
      revokeObjectURL: () => undefined,
    });
    renderUi(<App />);
    await user.type(screen.getByLabelText("Development token"), "dev-token-value");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    const box = await screen.findByRole("checkbox", { name: /مبيعات.csv/ });
    await waitFor(() => expect(box).toBeEnabled());
    await user.click(box);
    await user.type(screen.getByRole("textbox", { name: "Question" }), "sum المبيعات");
    await user.click(screen.getByRole("button", { name: "Calculate" }));
    expect((await screen.findAllByText(/30\.60/)).length).toBeGreaterThan(0);
    expect(screen.getAllByText("مبيعات.csv").some((node) => node.getAttribute("dir") === "auto")).toBe(
      true,
    );
    expect(screen.getAllByText("Not production qualified").length).toBeGreaterThan(0);
    expect(screen.getByText(/The planner ran/)).toBeInTheDocument();
    expect(screen.queryByText(/no usable extracted context/)).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some((call) => String(call[0]).endsWith("/answers"))).toBe(false);
    await user.click(screen.getByRole("button", { name: "Download CSV" }));
    expect(fetchMock.mock.calls.some((call) => String(call[0]).endsWith("/table.csv"))).toBe(true);
  });

  it("follows /calculate on the answer stream and loads the stored record", async () => {
    const user = userEvent.setup();
    const encoder = new TextEncoder();
    const fetchMock = workspaceFetch(async (url, init) => {
      if (url.endsWith("/answers") && init?.method === "POST") {
        const body = JSON.parse(String(init.body)) as { question: string; version_ids: string[] };
        expect(body.question).toBe("/calculate sum المبيعات");
        expect(body.version_ids).toEqual([version]);
        const payload = [
          JSON.stringify({ event: "queued", run_id: calculation, external_processing: false }),
          JSON.stringify({ event: "stage", stage: "calculating" }),
          JSON.stringify({
            event: "answer",
            mode: "spreadsheet_calculation",
            answer: completedRecord().record.answer,
            claims: [],
            scope_notice: "Descriptive aggregate from the authorized CSV version.",
            downloads: completedRecord().downloads,
            metadata: {
              calculation_id: calculation,
              calculation_status: "completed",
              model_called: false,
              production_qualified: false,
            },
          }),
          JSON.stringify({ event: "done", run_id: calculation }),
        ].join("\n");
        return new Response(encoder.encode(payload), {
          status: 200,
          headers: { "content-type": "application/x-ndjson" },
        });
      }
      if (url.endsWith(`/calculations/${calculation}`) && (!init?.method || init.method === "GET")) {
        return json({
          calculation_id: calculation,
          status: "completed",
          record: completedRecord().record,
          production_qualified: false,
        });
      }
      return undefined as unknown as Response;
    });
    vi.stubGlobal("fetch", fetchMock);
    renderUi(<App />);
    await user.type(screen.getByLabelText("Development token"), "dev-token-value");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    const box = await screen.findByRole("checkbox", { name: /مبيعات.csv/ });
    await waitFor(() => expect(box).toBeEnabled());
    await user.click(box);
    await user.type(screen.getByRole("textbox", { name: "Question" }), "/calculate sum المبيعات");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    expect((await screen.findAllByText(/30\.60/)).length).toBeGreaterThan(0);
    expect(fetchMock.mock.calls.some((call) => String(call[0]).endsWith(`/calculations/${calculation}`))).toBe(
      true,
    );
  });

  it("does not calculate a PDF selection", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/health/ready") return json({ generation: "disabled" });
      if (url === "/health/generation") return json({ mode: "fixture", production_qualified: false });
      if (url === "/api/workspaces") {
        return json({ workspaces: [{ id: workspace, name: "Synthetic workspace A" }] });
      }
      if (url.endsWith("/documents")) {
        return json({
          documents: [
            {
              id: "doc",
              filename: "تقرير.pdf",
              version_id: version,
              version_number: 1,
              byte_size: 40,
              sha256: "abc123abc123",
              status: "stored",
              ingestion_status: "extracted",
            },
          ],
        });
      }
      if (url.endsWith("/extraction")) return json({ status: "extracted", retrieval_ready: true, warnings: [] });
      return json({ detail: "missing" }, 404);
    });
    vi.stubGlobal("fetch", fetchMock);
    renderUi(<App />);
    await user.type(screen.getByLabelText("Development token"), "dev-token-value");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    const box = await screen.findByRole("checkbox", { name: /تقرير.pdf/ });
    await waitFor(() => expect(box).toBeEnabled());
    await user.click(box);
    expect(screen.getByRole("button", { name: "Calculate" })).toBeDisabled();
    await user.type(screen.getByRole("textbox", { name: "Question" }), "/calculate sum amount");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Calculations run on one ready CSV version.");
    expect(fetchMock.mock.calls.some((call) => String(call[0]).includes("/calculations"))).toBe(false);
    expect(fetchMock.mock.calls.some((call) => String(call[0]).endsWith("/answers"))).toBe(false);
  });
});

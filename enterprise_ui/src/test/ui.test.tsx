import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { readNdjson, uploadProblem } from "../api";
import { BidiText } from "../bidi";
import { readCalculationTables } from "../calculations/ResultTables";
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

  it("leaves formula text literal in a future result table", () => {
    const tables = readCalculationTables({
      tables: [
        {
          id: "t1",
          title: "Totals المجموع",
          columns: ["Item", "Value"],
          rows: [["=1+1", "2"]],
        },
      ],
    });
    const entry: ChatEntry = {
      id: "calc",
      role: "assistant",
      kind: "answer",
      events: [{ ...answer, calculations: { tables } }],
    };
    renderUi(<AnswerCard entry={entry} onPreview={() => undefined} />);
    expect(screen.getByText("=1+1")).toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument();
  });
});

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
    expect(uploadProblem({ name: "notes.md", size: 10 })).toBeNull();
    expect(uploadProblem({ name: "sheet.xlsx", size: 10 })).toBe("type");
    expect(uploadProblem({ name: "big.pdf", size: 5_000_001 })).toBe("size");
  });
});

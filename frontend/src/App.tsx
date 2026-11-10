import { lazy, Suspense, useEffect, useState } from "react";

const ModelEvaluation = lazy(() => import("./ModelEvaluation"));

import {
  ShieldCheck,
  UploadCloud,
  FileSearch,
  Activity,
  Fingerprint,
  Trash2,
  AlertTriangle,
  ArrowUpRight,
  LoaderCircle,
  ScanFace,
} from "lucide-react";

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const apiFetch = (
  input: RequestInfo | URL,
  init: RequestInit = {},
) => fetch(input, { ...init, credentials: "include" });

type Analysis = {
  analysis_id: string;

  status: string;

  stage: string;

  progress: number;

  filename: string;

  size_bytes: number;

  sha256: string;

  result?: any;

  error?: string;

  created_at?: string;
};

type HistoryResponse = {
  total: number;

  analyses: Analysis[];
};

const stages = ["validating", "metadata", "sampling", "inference", "completed"];

function formatBytes(n: number) {
  if (!n) return "â€”";

  const units = ["B", "KB", "MB", "GB"];

  let i = 0;

  let v = n;

  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;

    i++;
  }

  return `${v.toFixed(i ? 1 : 0)} ${units[i]}`;
}

function shortHash(hash: string) {
  return hash ? `${hash.slice(0, 16)}â€¦${hash.slice(-8)}` : "Pending";
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);

  const [consent, setConsent] = useState(false);

  const [busy, setBusy] = useState(false);

  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [compareFrameA, setCompareFrameA] = useState<string>("");
  const [compareFrameB, setCompareFrameB] = useState<string>("");
  const [selectedFrameModel, setSelectedFrameModel] = useState<string>("");

  const [error, setError] = useState("");

  const [tab, setTab] = useState("overview");

  const tabs = ["overview", "evidence", "stability", "provenance"] as const;

  const [drag, setDrag] = useState(false);

  const [history, setHistory] = useState<Analysis[]>([]);

  const [historyLoading, setHistoryLoading] = useState(true);

  const [historyError, setHistoryError] = useState("");

  const [historySearch, setHistorySearch] = useState("");
  const [reportBusy, setReportBusy] = useState(false);
  const [evidenceBusy, setEvidenceBusy] = useState(false);

  const [stability, setStability] = useState<any>(null);

  const [stabilityBusy, setStabilityBusy] = useState(false);

  const [stabilityError, setStabilityError] = useState("");

  async function refreshHistory() {
    setHistoryLoading(true);

    setHistoryError("");

    try {
      const res = await apiFetch(`${API}/api/v1/analyses`);

      if (!res.ok) throw new Error(await res.text());

      const data: HistoryResponse = await res.json();

      const analyses = Array.isArray(data.analyses) ? data.analyses : [];

      analyses.sort((a, b) => {
        const dateA = a.created_at ? Date.parse(a.created_at) : 0;

        const dateB = b.created_at ? Date.parse(b.created_at) : 0;

        return dateB - dateA;
      });

      setHistory(analyses);
    } catch (e: any) {
      setHistoryError(e.message || "Could not load analysis history.");
    } finally {
      setHistoryLoading(false);
    }
  }

  async function refresh(id: string) {
    const res = await apiFetch(`${API}/api/v1/analyses/${id}`);

    if (!res.ok) throw new Error(await res.text());

    const data: Analysis = await res.json();

    setAnalysis(data);

    setStability(data.result?.stability || null);

    // Keep the sidebar history in sync as soon as an analysis reaches a

    // terminal state, without requiring the user to press Refresh history.

    if (["completed", "failed"].includes(data.status)) {
      void refreshHistory();
    }

    return data;
  }

  async function openSavedAnalysis(id: string) {
    setError("");

    setStabilityError("");

    try {
      await refresh(id);

      setTab("overview");
    } catch (e: any) {
      setError(e.message || "Could not open the saved analysis.");

      await refreshHistory();
    }
  }

  async function deleteSavedAnalysis(id: string) {
    const saved = history.find((item) => item.analysis_id === id);

    if (!saved) return;

    try {
      const res = await apiFetch(`${API}/api/v1/analyses/${id}`, {
        method: "DELETE",
      });

      if (!res.ok) throw new Error("Could not delete this analysis.");

      if (analysis?.analysis_id === id) {
        setAnalysis(null);

        setStability(null);

        setStabilityError("");

        setFile(null);

        setConsent(false);

        setTab("overview");
      }

      await refreshHistory();
    } catch (e: any) {
      setHistoryError(e.message || "Delete failed.");
    }
  }

  useEffect(() => {
    refreshHistory();
  }, []);

  useEffect(() => {
    if (
      !analysis?.analysis_id ||
      ["completed", "failed"].includes(analysis.status)
    ) {
      return;
    }

    const timer = window.setInterval(() => {
      refresh(analysis.analysis_id).catch((e) => setError(String(e)));
    }, 1200);

    return () => window.clearInterval(timer);
  }, [analysis?.analysis_id, analysis?.status]);

  async function submit() {
    if (!file || !consent) return;

    setBusy(true);

    setError("");

    setAnalysis(null);

    setStability(null);

    setStabilityError("");

    try {
      const body = new FormData();

      body.append("video", file);

      body.append("consent", "true");

      const res = await apiFetch(`${API}/api/v1/analyses`, {
        method: "POST",

        body,
      });

      if (!res.ok) {
        throw new Error((await res.json()).detail || "Upload failed");
      }

      const data = await res.json();

      await refresh(data.analysis_id);

      await refreshHistory();
    } catch (e: any) {
      setError(e.message || "Could not connect to the API.");
    } finally {
      setBusy(false);
    }
  }

  async function downloadReport() {
    if (!analysis?.analysis_id || analysis.status !== "completed") return;

    setReportBusy(true);
    setError("");

    try {
      const response = await apiFetch(
        `${API}/api/v1/analyses/${encodeURIComponent(analysis.analysis_id)}/report`,
        { headers: { Accept: "application/pdf" } },
      );

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || "PDF generation failed.");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `veriframe_report_${analysis.analysis_id.slice(0, 8)}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e.message || "Could not download PDF.");
    } finally {
      setReportBusy(false);
    }
  }

  async function downloadEvidencePackage() {
    if (!analysis?.analysis_id) return;

    setEvidenceBusy(true);
    setError("");

    try {
      const response = await apiFetch(
        `${API}/api/v1/analyses/${analysis.analysis_id}/evidence-package`,
      );

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || "Evidence package generation failed.");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");

      link.href = url;
      link.download = `veriframe_evidence_${analysis.analysis_id.slice(0, 8)}.zip`;

      document.body.appendChild(link);
      link.click();
      link.remove();

      window.URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e.message || "Could not download evidence package.");
    } finally {
      setEvidenceBusy(false);
    }
  }

  async function deleteAnalysis() {
    if (!analysis) return;

    try {
      const res = await apiFetch(
        `${API}/api/v1/analyses/${analysis.analysis_id}`,

        { method: "DELETE" },
      );

      if (!res.ok) throw new Error("Delete failed");

      setAnalysis(null);

      setStability(null);

      setStabilityError("");

      setFile(null);

      setConsent(false);

      setTab("overview");

      await refreshHistory();
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function runStabilityCheck() {
    if (!analysis?.analysis_id) return;

    setStabilityBusy(true);

    setStabilityError("");

    try {
      const res = await apiFetch(
        `${API}/api/v1/analyses/${analysis.analysis_id}/stability`,

        { method: "POST" },
      );

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Stability check failed.");
      }

      setStability(data);
    } catch (e: any) {
      setStabilityError(e.message || "Could not connect to the stability API.");
    } finally {
      setStabilityBusy(false);
    }
  }

  const result = analysis?.result;

  const normalizedHistorySearch = historySearch.trim().toLowerCase();
  const filteredHistory: Analysis[] = history.filter((item: Analysis) => {
    if (!normalizedHistorySearch) return true;
    return (
      (item.filename || "").toLowerCase().includes(normalizedHistorySearch) ||
      (item.analysis_id || "")
        .toLowerCase()
        .includes(normalizedHistorySearch) ||
      (item.status || "").toLowerCase().includes(normalizedHistorySearch)
    );
  });

  // Accept the common response keys while keeping the UI safe when the
  // stability endpoint returns no per-transformation results.
  const stabilityTests: any[] = Array.isArray(stability?.tests)
    ? stability.tests
    : Array.isArray(stability?.results)
      ? stability.results
      : Array.isArray(stability?.perturbations)
        ? stability.perturbations
        : [];

  const stabilityCompleted = stabilityTests.filter(
    (test: any) => test?.status === "completed",
  );

  const largestStabilityChange = stabilityCompleted
    .filter((test: any) => {
      const delta = Number(test?.absolute_score_delta);
      return test?.absolute_score_delta != null && Number.isFinite(delta);
    })
    .reduce((largest: any, test: any) => {
      if (
        !largest ||
        Number(test.absolute_score_delta) > Number(largest.absolute_score_delta)
      ) {
        return test;
      }
      return largest;
    }, null);

  const currentStage = analysis?.stage || "queued";

  const stageIndex = stages.indexOf(currentStage);

  const evidence = result?.evidence;

  const gend = result?.committee?.gend;
  const xception = result?.committee?.xception;
  const effort = result?.committee?.effort;
  const fakestormer = result?.committee?.fakestormer;

  const gendReady = gend?.status === "ok";
  const xceptionReady = xception?.status === "ok";
  const effortReady = effort?.status === "ok";
  const fakestormerReady = fakestormer?.status === "ok";

  const fakestormerTemporalValues: number[] = Array.isArray(
    fakestormer?.temporal_localization_values,
  )
    ? fakestormer.temporal_localization_values
        .map((value: any) => Number(value))
        .filter((value: number) => Number.isFinite(value))
    : [];
  const fakestormerSampleIndices: number[] = Array.isArray(
    fakestormer?.sampled_frame_indices,
  )
    ? fakestormer.sampled_frame_indices.map((value: any) => Number(value))
    : [];
  const fakestormerFps = Number(fakestormer?.source_fps);
  const fakestormerFinalValue =
    typeof fakestormer?.temporal_localization_final === "number"
      ? fakestormer.temporal_localization_final
      : null;

  const inferenceReady =
    gendReady || xceptionReady || effortReady || fakestormerReady;

  const activeModelCount =
    Number(gendReady) +
    Number(xceptionReady) +
    Number(effortReady) +
    Number(fakestormerReady);

  const gendScoreText =
    typeof gend?.class1_manipulated_score_mean === "number"
      ? gend.class1_manipulated_score_mean.toFixed(4)
      : "Unavailable";

  const xceptionScoreText =
    typeof xception?.class1_manipulated_score_mean === "number"
      ? xception.class1_manipulated_score_mean.toFixed(4)
      : "Unavailable";
  const calibratedProbabilities =
    result?.committee?.calibrated_probabilities || {};

  const gendCalibratedText =
    typeof calibratedProbabilities.gend === "number"
      ? `${(calibratedProbabilities.gend * 100).toFixed(2)}%`
      : "Unavailable";

  const xceptionCalibratedText =
    typeof calibratedProbabilities.xception === "number"
      ? `${(calibratedProbabilities.xception * 100).toFixed(2)}%`
      : "Unavailable";
  const effortScoreText =
    typeof effort?.class1_manipulated_score_mean === "number"
      ? effort.class1_manipulated_score_mean.toFixed(4)
      : "Unavailable";
  const modelAgreement =
    gendReady && xceptionReady && gend.output != null && xception.output != null
      ? gend.output === xception.output
      : null;

  // Use GenD frames by default; fall back to Xception if GenD is unavailable.
  // This prevents duplicate frame-index collisions between the two models.
  // Build the detector list from models that actually returned frame scores.
  const allFrameScores = Array.isArray(result?.frames) ? result.frames : [];

  const frameModelLabels: Record<string, string> = {
    gend_clip_l14: "GenD CLIP ViT-L/14",
    xception_custom: "Custom DeepfakeBench Xception",
    effort: "Effort",
  };

  const preferredFrameModels = ["gend_clip_l14", "xception_custom", "effort"];

  const availableFrameModels = preferredFrameModels.filter((modelId) =>
    allFrameScores.some((frame: any) => {
      const score = frame.class1_manipulated_score ?? frame.manipulated_score;

      return (
        frame.model_id === modelId &&
        score !== null &&
        score !== undefined &&
        Number.isFinite(Number(score))
      );
    }),
  );

  const frameModelId = availableFrameModels.includes(selectedFrameModel)
    ? selectedFrameModel
    : availableFrameModels[0] || "";

  const frameModelLabel =
    frameModelLabels[frameModelId] || "No frame model available";

  const frameScores = allFrameScores
    .filter((frame: any) => frame.model_id === frameModelId)
    .map((frame: any) => ({
      ...frame,
      class1_manipulated_score:
        frame.class1_manipulated_score ?? frame.manipulated_score,
    }));

  const highestScoringFrame = frameScores.reduce((highest: any, frame: any) => {
    const score = Number(frame.class1_manipulated_score);

    if (!Number.isFinite(score)) return highest;

    if (!highest || score > Number(highest.class1_manipulated_score)) {
      return frame;
    }

    return highest;
  }, null);

  const topSuspiciousFrames = [...frameScores]
    .filter((frame: any) =>
      Number.isFinite(Number(frame.class1_manipulated_score)),
    )
    .sort(
      (a: any, b: any) =>
        Number(b.class1_manipulated_score) - Number(a.class1_manipulated_score),
    )
    .slice(0, 5);

  const selectedFrameA =
    frameScores.find(
      (frame: any) => String(frame.frame_index) === compareFrameA,
    ) ?? null;

  const selectedFrameB =
    frameScores.find(
      (frame: any) => String(frame.frame_index) === compareFrameB,
    ) ?? null;
  function formatDate(value?: string) {
    if (!value) return "Date unavailable";

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) return "Date unavailable";

    return date.toLocaleString([], {
      year: "numeric",

      month: "short",

      day: "numeric",

      hour: "2-digit",

      minute: "2-digit",
    });
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ScanFace size={22} />
          </div>

          <div>
            <strong>VeriFrame</strong>

            <span>VIDEO FORENSICS</span>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <button
          className={`nav-item ${tab === "overview" ? "active" : ""}`}
          onClick={() => setTab("overview")}
        >
          <Activity size={17} /> Analysis workspace
        </button>

        <button
          className={`nav-item ${tab === "evidence" ? "active" : ""}`}
          onClick={() => setTab("evidence")}
        >
          <FileSearch size={17} /> Evidence ledger
        </button>

        <button
          className={`nav-item ${tab === "stability" ? "active" : ""}`}
          onClick={() => setTab("stability")}
        >
          <Activity size={17} /> Stability panel
        </button>

        <button
          className={`nav-item ${tab === "provenance" ? "active" : ""}`}
          onClick={() => setTab("provenance")}
        >
          <Fingerprint size={17} /> Provenance
        </button>

        <button
          className={`nav-item ${tab === "model-evaluation" ? "active" : ""}`}
          onClick={() => setTab("model-evaluation")}
        >
          <Activity size={17} /> Model Evaluation
        </button>

        <div style={{ marginTop: 24, minHeight: 0 }}>
          <div className="workspace-label" style={{ marginBottom: 10 }}>
            PREVIOUS ANALYSES ({history.length})
          </div>

          <button
            className="text-button"
            onClick={refreshHistory}
            disabled={historyLoading}
            style={{ marginBottom: 10 }}
          >
            {historyLoading ? "Refreshingâ€¦" : "Refresh history"}
          </button>

          <input
            type="search"
            value={historySearch}
            onChange={(event) => setHistorySearch(event.target.value)}
            placeholder="Search video historyâ€¦"
            aria-label="Search previous analyses"
            style={{
              width: "100%",

              boxSizing: "border-box",

              marginBottom: 10,

              padding: "9px 10px",

              borderRadius: 8,

              border: "1px solid rgba(148, 163, 184, 0.25)",

              background: "rgba(15, 23, 42, 0.55)",

              color: "inherit",

              fontSize: 12,

              outline: "none",
            }}
          />

          {historyError && (
            <div className="error-box" style={{ fontSize: 12, padding: 8 }}>
              {historyError}
            </div>
          )}

          {!historyLoading && history.length === 0 && !historyError && (
            <div className="muted" style={{ fontSize: 12, padding: "6px 0" }}>
              No saved analyses yet.
            </div>
          )}

          {history.length > 0 && filteredHistory.length === 0 && (
            <div className="muted" style={{ fontSize: 12, padding: "6px 0" }}>
              No matching videos.
            </div>
          )}

          <div style={{ maxHeight: 300, overflowY: "auto" }}>
            {filteredHistory.map((item) => (
              <div
                key={item.analysis_id}
                style={{
                  display: "flex",

                  alignItems: "center",

                  gap: 6,

                  padding: "8px 4px",

                  borderBottom: "1px solid rgba(148, 163, 184, 0.14)",
                }}
              >
                <button
                  onClick={() => openSavedAnalysis(item.analysis_id)}
                  title={item.filename}
                  style={{
                    flex: 1,

                    minWidth: 0,

                    textAlign: "left",

                    background: "transparent",

                    border: 0,

                    color: "inherit",

                    cursor: "pointer",

                    padding: 4,
                  }}
                >
                  <strong
                    style={{
                      display: "block",

                      fontSize: 12,

                      overflow: "hidden",

                      textOverflow: "ellipsis",

                      whiteSpace: "nowrap",
                    }}
                  >
                    {item.filename || "Untitled video"}
                  </strong>

                  <span
                    className="muted"
                    style={{ display: "block", fontSize: 10, marginTop: 3 }}
                  >
                    {(item.status || "unknown").toUpperCase()} Â·{" "}
                    {formatBytes(item.size_bytes)}
                  </span>

                  <span
                    className="muted"
                    style={{ display: "block", fontSize: 10, marginTop: 2 }}
                  >
                    {formatDate(item.created_at)}
                  </span>
                </button>

                <button
                  className="text-button"
                  title={`Delete ${item.filename || "analysis"}`}
                  aria-label={`Delete ${item.filename || "analysis"}`}
                  onClick={() => deleteSavedAnalysis(item.analysis_id)}
                  style={{ padding: 5 }}
                >
                  <Trash2 size={13} />
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="secure-dot" /> Private analysis workspace
          <span className="muted">Starter build Â· local API</span>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <div className="eyebrow">FORENSIC WORKSPACE / V1</div>

            <h1>Video analysis</h1>
          </div>

          <div className="top-actions">
            <span className="environment">
              <span /> DEMO ENVIRONMENT
            </span>

            <div className="avatar">VF</div>
          </div>
        </header>

        <div className="content">
          <div className="notice">
            <AlertTriangle size={17} />

            <div>
              <strong>Research prototype</strong>

              <span>
                {inferenceReady
                  ? "GenD and custom Xception provide calibrated manipulated probabilities using frozen calibration parameters. The final verdict is based on agreement between both primary models; disagreement or an unavailable primary score produces INCONCLUSIVE. Effort provides supporting evidence, while FakeSTormer reports temporal-localization evidence separately. Results should be interpreted within the scope of the available validation."
                  : "The detector inference bridges run in separate Python environments. If no bridge is configured, the upload will still be recorded but no model score will be available."}
              </span>
            </div>
          </div>

          {!analysis && tab !== "model-evaluation" && (
            <section className="upload-card">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">NEW INVESTIGATION</span>

                  <h2>Analyze a video</h2>

                  <p>
                    Submit a video to start the forensic processing workflow.
                  </p>
                </div>

                <div className="upload-icon">
                  <UploadCloud size={22} />
                </div>
              </div>

              <div
                className={`dropzone ${drag ? "dragging" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();

                  setDrag(true);
                }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => {
                  e.preventDefault();

                  setDrag(false);

                  const f = e.dataTransfer.files?.[0];

                  if (f) setFile(f);
                }}
                onClick={() => document.getElementById("video-input")?.click()}
              >
                <input
                  id="video-input"
                  type="file"
                  accept=".mp4,.mov,.webm,.avi,video/mp4,video/quicktime,video/webm,video/x-msvideo"
                  hidden
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                />

                <div className="cloud-icon">
                  <UploadCloud size={25} />
                </div>

                <strong>
                  {file ? file.name : "Drop your video here or browse files"}
                </strong>

                <span>
                  {file
                    ? `${formatBytes(file.size)} Â· ${file.type || "video file"}`
                    : "MP4, MOV, WEBM, AVI Â· maximum 250 MB"}
                </span>

                {file && (
                  <button
                    className="text-button"
                    onClick={(e) => {
                      e.stopPropagation();

                      setFile(null);
                    }}
                  >
                    Remove file
                  </button>
                )}
              </div>

              <label className="consent">
                <input
                  type="checkbox"
                  checked={consent}
                  onChange={(e) => setConsent(e.target.checked)}
                />

                <span>
                  I have permission to analyze this video and understand that
                  this research prototype is not proof of authenticity.
                </span>
              </label>

              <div className="upload-footer">
                <span>
                  <ShieldCheck size={15} /> SHA-256 hashing enabled Â· local
                  starter storage
                </span>

                <button
                  className="primary-button"
                  disabled={!file || !consent || busy}
                  onClick={submit}
                >
                  {busy ? (
                    <>
                      <LoaderCircle className="spin" size={16} /> Uploadingâ€¦
                    </>
                  ) : (
                    <>
                      Start analysis <ArrowUpRight size={16} />
                    </>
                  )}
                </button>
              </div>

              {error && <div className="error-box">{error}</div>}
            </section>
          )}

          {tab === "model-evaluation" && (
            <Suspense
              fallback={
                <div className="panel empty-state">
                  <LoaderCircle className="spin" size={23} />
                  <strong>Loading model evaluation...</strong>
                </div>
              }
            >
              <ModelEvaluation />
            </Suspense>
          )}

          {analysis && tab !== "model-evaluation" && (
            <>
              <div className="analysis-header">
                <div>
                  <div className="eyebrow">
                    ANALYSIS ID Â·{" "}
                    {analysis.analysis_id.slice(0, 8).toUpperCase()}
                  </div>

                  <h2>{analysis.filename}</h2>

                  <p>
                    {formatBytes(analysis.size_bytes)} Â· SHA-256{" "}
                    {shortHash(analysis.sha256)}
                  </p>
                </div>

                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <button
                    className="secondary-button"
                    onClick={downloadReport}
                    disabled={reportBusy || analysis.status !== "completed"}
                  >
                    {reportBusy ? (
                      <>
                        <LoaderCircle className="spin" size={15} />
                        Generating PDF...
                      </>
                    ) : (
                      <>
                        <FileSearch size={15} /> Download PDF Report
                      </>
                    )}
                  </button>
                  <button
                    className="secondary-button"
                    onClick={downloadEvidencePackage}
                    disabled={evidenceBusy || analysis.status !== "completed"}
                  >
                    {evidenceBusy ? (
                      <>
                        <LoaderCircle className="spin" size={15} />
                        Generating ZIP...
                      </>
                    ) : (
                      <>
                        <FileSearch size={15} /> Download Evidence Package
                      </>
                    )}
                  </button>
                  <button className="secondary-button" onClick={deleteAnalysis}>
                    <Trash2 size={15} /> Delete analysis
                  </button>
                </div>
              </div>

              {analysis.status !== "completed" &&
                analysis.status !== "failed" && (
                  <section className="panel progress-panel">
                    <div className="panel-title">
                      <div>
                        <h3>Processing pipeline</h3>

                        <p>
                          Analysis runs asynchronously; progress updates from
                          the API.
                        </p>
                      </div>

                      <span className="status-pill running">
                        <LoaderCircle className="spin" size={13} />{" "}
                        {analysis.status}
                      </span>
                    </div>

                    <div className="progress-track">
                      <div style={{ width: `${analysis.progress || 0}%` }} />
                    </div>

                    <div className="progress-meta">
                      <span>{currentStage.replace(/_/g, " ")}</span>

                      <strong>{analysis.progress || 0}%</strong>
                    </div>

                    <div className="stage-grid">
                      {stages.map((s, i) => (
                        <div
                          key={s}
                          className={`stage ${stageIndex >= i ? "done" : ""}`}
                        >
                          <span>
                            {stageIndex > i
                              ? "âœ“"
                              : String(i + 1).padStart(2, "0")}
                          </span>

                          {s}
                        </div>
                      ))}
                    </div>
                  </section>
                )}

              {analysis.status === "failed" && (
                <div className="error-box">
                  {analysis.error || "Analysis failed."}
                </div>
              )}

              {result && (
                <>
                  <section
                    className={`verdict-card verdict-${(result.outcome || "INCONCLUSIVE").toLowerCase()}`}
                  >
                    <div className="verdict-icon">
                      <AlertTriangle size={23} />
                    </div>

                    <div className="verdict-main">
                      <div className="eyebrow">CURRENT ASSESSMENT</div>

                      <h2>{result.outcome || "INCONCLUSIVE"}</h2>

                      <p>{result.outcome_note}</p>

                      <div className="verdict-tags">
                        <span>GenD manipulated score: {gendScoreText}</span>

                        <span>
                          Xception manipulated score: {xceptionScoreText}
                        </span>

                        <span>
                          Model agreement:{" "}
                          {modelAgreement === null
                            ? "Not comparable"
                            : modelAgreement
                              ? "Same threshold-based output"
                              : "Different threshold-based outputs"}
                        </span>

                        <span>
                          GenD calibrated fake probability: {gendCalibratedText}
                        </span>
                        <span>
                          Xception calibrated fake probability:{" "}
                          {xceptionCalibratedText}
                        </span>

                        <span>Conformal set: not run</span>
                      </div>
                    </div>

                    <div className="verdict-side">
                      <span className="status-pill neutral">
                        {activeModelCount > 0
                          ? `${activeModelCount} models ran`
                          : "Inference unavailable"}
                      </span>

                      <small>
                        {inferenceReady
                          ? `GenD: ${gendReady ? gend.sampled_frame_count : "â€”"} Â· Xception: ${xceptionReady ? xception.sampled_frame_count : "â€”"} Â· Effort: ${effortReady ? effort.sampled_frame_count : "â€”"} Â· FakeSTormer: ${fakestormerReady ? fakestormer.sampled_frame_count : "â€”"} frames`
                          : "No model score"}
                      </small>
                    </div>
                  </section>

                  <div className="tabs">
                    {tabs.map((t) => (
                      <button
                        key={t}
                        className={tab === t ? "selected" : ""}
                        onClick={() => setTab(t)}
                      >
                        {t === "overview"
                          ? "Overview"
                          : t === "evidence"
                            ? "Evidence ledger"
                            : t === "stability"
                              ? "Stability panel"
                              : "Provenance"}
                      </button>
                    ))}
                  </div>

                  {tab === "overview" && (
                    <div className="grid-two">
                      <section className="panel">
                        <div className="panel-title">
                          <div>
                            <h3>Detector status</h3>

                            <p>Current model integration</p>
                          </div>

                          <span className="mini-tag">
                            {activeModelCount} active
                          </span>
                        </div>

                        {(result.committee?.models || []).map((m: any) => (
                          <div className="model-row" key={m.id}>
                            <div className="model-avatar">
                              {m.id.includes("xception")
                                ? "X"
                                : m.id.includes("clip")
                                  ? "CL"
                                  : m.id.includes("dino")
                                    ? "DN"
                                    : "CNN"}
                            </div>

                            <div className="model-info">
                              <strong>{m.id}</strong>

                              <span>{m.role || "Detector adapter"}</span>
                            </div>

                            <span className="status-pill neutral">
                              {m.status}
                            </span>
                          </div>
                        ))}

                        <div className="empty-inline">
                          {inferenceReady
                            ? `GenD: ${gendReady ? "active" : "unavailable"} Â· Xception: ${xceptionReady ? "active" : "unavailable"} Â· Effort: ${effortReady ? "active" : "unavailable"} Â· FakeSTormer: ${fakestormerReady ? "active" : "unavailable"}.`
                            : result.committee?.reason ||
                              "Configure the detector inference bridges to enable scoring."}
                        </div>
                      </section>

                      <section className="panel">
                        <div className="panel-title">
                          <div>
                            <h3>Evidence summary</h3>

                            <p>Separated evidence channels</p>
                          </div>

                          <FileSearch size={18} />
                        </div>

                        <div className="metric-row">
                          <span>Model evidence</span>

                          <strong>
                            {inferenceReady
                              ? `GenD: ${gendScoreText} Â· Xception: ${xceptionScoreText} Â· Effort: ${effortScoreText} (uncalibrated)`
                              : "Not available"}
                          </strong>
                        </div>

                        <div className="metric-row">
                          <span>Media / provenance</span>

                          <strong>SHA-256 recorded</strong>
                        </div>

                        <div className="metric-row">
                          <span>Quality evidence</span>

                          <strong>Not measured</strong>
                        </div>

                        <div className="metric-row">
                          <span>Sampled frame timeline</span>
                          <strong>
                            {inferenceReady
                              ? `GenD: ${gendReady ? gend.sampled_frame_count : "â€”"} Â· Xception: ${xceptionReady ? xception.sampled_frame_count : "â€”"} Â· Effort: ${effortReady ? effort.sampled_frame_count : "â€”"} Â· FakeSTormer: ${fakestormerReady ? fakestormer.sampled_frame_count : "â€”"}`
                              : "Not generated"}
                          </strong>
                        </div>

                        <div className="metric-row">
                          <span>
                            FakeSTormer temporal localization (uncalibrated)
                          </span>
                          <strong>
                            {fakestormerReady
                              ? `Final value: ${
                                  typeof fakestormer.temporal_localization_final ===
                                  "number"
                                    ? fakestormer.temporal_localization_final.toFixed(
                                        4,
                                      )
                                    : "Unavailable"
                                } Â· Label: ${fakestormer.official_label || "Unavailable"} Â· Threshold: ${
                                  typeof fakestormer.threshold === "number"
                                    ? fakestormer.threshold.toFixed(2)
                                    : "Unavailable"
                                }`
                              : "Not available"}
                          </strong>
                        </div>
                      </section>
                    </div>
                  )}

                  {inferenceReady && (
                    <section className="panel frame-panel">
                      {availableFrameModels.length > 1 && (
                        <div className="frame-model-selector">
                          <label htmlFor="frame-model-select">
                            <span className="eyebrow">
                              FRAME ANALYSIS MODEL
                            </span>
                            <select
                              id="frame-model-select"
                              value={frameModelId}
                              onChange={(e) => {
                                setSelectedFrameModel(e.target.value);
                                setCompareFrameA("");
                                setCompareFrameB("");
                              }}
                            >
                              {availableFrameModels.map((modelId) => (
                                <option key={modelId} value={modelId}>
                                  {frameModelLabels[modelId]}
                                </option>
                              ))}
                            </select>
                          </label>
                          <p>
                            Frame rankings, scores and comparisons are shown for
                            the selected detector.
                          </p>
                        </div>
                      )}
                      {highestScoringFrame && (
                        <div className="highest-frame-card">
                          <div>
                            <span className="eyebrow">
                              HIGHEST-SCORING SAMPLED FRAME
                            </span>
                            <h3>Frame {highestScoringFrame.frame_index}</h3>
                            <p>
                              Timestamp:{" "}
                              {Number(
                                highestScoringFrame.timestamp_seconds,
                              ).toFixed(2)}{" "}
                              seconds
                            </p>
                          </div>
                          <button
                            className="highest-frame-jump"
                            onClick={() => {
                              const target = document.getElementById(
                                `frame-card-${frameModelId}-${highestScoringFrame.frame_index}`,
                              ) as HTMLDetailsElement | null;

                              if (target) {
                                target.open = true;
                                target.scrollIntoView({
                                  behavior: "smooth",
                                  block: "center",
                                });
                              }
                            }}
                          >
                            View frame in timeline â†’
                          </button>

                          <div className="highest-frame-score">
                            {Number(
                              highestScoringFrame.class1_manipulated_score,
                            ).toFixed(4)}
                            <span>
                              {frameModelLabel} Â· Uncalibrated manipulated score
                            </span>
                          </div>
                        </div>
                      )}
                      <div className="panel-title">
                        <div>
                          <h3>Frame score timeline</h3>
                          <div className="top-frames-panel">
                            <div className="top-frames-heading">
                              <div>
                                <span className="eyebrow">FRAME RANKING</span>
                                <h3>Top 5 highest-scoring frames</h3>
                                <p>Ranked by uncalibrated manipulated score.</p>
                              </div>
                              <span className="top-frames-count">
                                {topSuspiciousFrames.length} frames
                              </span>
                            </div>

                            <div className="top-frames-grid">
                              {topSuspiciousFrames.map(
                                (frame: any, index: number) => (
                                  <div
                                    className="top-frame-card"
                                    key={`${frame.frame_index}-${frame.timestamp_seconds}`}
                                  >
                                    <div className="top-frame-rank">
                                      #{index + 1}
                                    </div>

                                    <img
                                      src={`${API}/api/v1/analyses/${encodeURIComponent(
                                        analysis!.analysis_id,
                                      )}/thumbnail?timestamp=${Number(
                                        frame.timestamp_seconds || 0,
                                      )}`}
                                      alt={`Frame ${frame.frame_index}`}
                                      loading="lazy"
                                    />

                                    <div className="top-frame-info">
                                      <strong>Frame {frame.frame_index}</strong>
                                      <span>
                                        {Number(
                                          frame.timestamp_seconds,
                                        ).toFixed(2)}{" "}
                                        seconds
                                      </span>
                                      <span className="top-frame-score">
                                        {frameModelLabel}:{" "}
                                        {Number(
                                          frame.class1_manipulated_score,
                                        ).toFixed(4)}
                                      </span>
                                    </div>

                                    <button
                                      className="highest-frame-jump"
                                      onClick={() => {
                                        const target = document.getElementById(
                                          `frame-card-${frameModelId}-${frame.frame_index}`,
                                        ) as HTMLDetailsElement | null;

                                        if (target) {
                                          target.open = true;
                                          target.scrollIntoView({
                                            behavior: "smooth",
                                            block: "center",
                                          });
                                        }
                                      }}
                                    >
                                      View in timeline â†’
                                    </button>
                                  </div>
                                ),
                              )}
                            </div>
                          </div>

                          <p>
                            Uncalibrated manipulated score per sampled frame.
                            These scores are not probabilities.
                          </p>
                        </div>

                        <span className="mini-tag">
                          {frameScores.length} frames
                        </span>
                      </div>
                      <div className="frame-comparison-panel">
                        <div className="frame-comparison-heading">
                          <div>
                            <span className="eyebrow">FORENSIC REVIEW</span>
                            <h3>Frame comparison</h3>
                            <p>Compare two sampled frames side by side.</p>
                          </div>
                        </div>

                        <div className="frame-comparison-selectors">
                          <label>
                            First frame
                            <select
                              value={compareFrameA}
                              onChange={(e) => setCompareFrameA(e.target.value)}
                            >
                              <option value="">Select first frame</option>
                              {frameScores.map((frame: any) => (
                                <option
                                  key={`a-${frame.frame_index}`}
                                  value={String(frame.frame_index)}
                                >
                                  Frame {frame.frame_index} â€”{" "}
                                  {Number(frame.timestamp_seconds).toFixed(2)}s
                                </option>
                              ))}
                            </select>
                          </label>

                          <label>
                            Second frame
                            <select
                              value={compareFrameB}
                              onChange={(e) => setCompareFrameB(e.target.value)}
                            >
                              <option value="">Select second frame</option>
                              {frameScores.map((frame: any) => (
                                <option
                                  key={`b-${frame.frame_index}`}
                                  value={String(frame.frame_index)}
                                >
                                  Frame {frame.frame_index} â€”{" "}
                                  {Number(frame.timestamp_seconds).toFixed(2)}s
                                </option>
                              ))}
                            </select>
                          </label>
                        </div>

                        {selectedFrameA && selectedFrameB && (
                          <div className="frame-comparison-grid">
                            {[selectedFrameA, selectedFrameB].map(
                              (frame: any, index: number) => (
                                <div
                                  className="frame-comparison-card"
                                  key={`${index}-${frame.frame_index}`}
                                >
                                  <h4>
                                    {index === 0
                                      ? "First frame"
                                      : "Second frame"}
                                  </h4>

                                  <img
                                    src={`${API}/api/v1/analyses/${encodeURIComponent(
                                      analysis!.analysis_id,
                                    )}/thumbnail?timestamp=${Number(
                                      frame.timestamp_seconds || 0,
                                    )}`}
                                    alt={`Frame ${frame.frame_index}`}
                                    loading="lazy"
                                  />

                                  <div className="frame-comparison-details">
                                    <strong>Frame {frame.frame_index}</strong>
                                    <span>
                                      Timestamp:{" "}
                                      {Number(frame.timestamp_seconds).toFixed(
                                        2,
                                      )}{" "}
                                      seconds
                                    </span>
                                    <span>
                                      Uncalibrated manipulated score:{" "}
                                      {Number(
                                        frame.class1_manipulated_score,
                                      ).toFixed(4)}
                                    </span>
                                  </div>
                                </div>
                              ),
                            )}
                          </div>
                        )}

                        <p className="frame-comparison-disclaimer">
                          Scores are uncalibrated model outputs, not
                          probabilities or confirmation of manipulation.
                        </p>
                      </div>

                      <div className="frame-list">
                        {frameScores.map((f: any) => {
                          const rawScore = Number(
                            f.class1_manipulated_score || 0,
                          );

                          const score = Number.isFinite(rawScore)
                            ? Math.max(0, Math.min(1, rawScore))
                            : 0;

                          // Display-only bands, not validated detection thresholds.
                          const level =
                            score >= 0.7
                              ? "higher"
                              : score >= 0.4
                                ? "middle"
                                : "lower";

                          return (
                            <details
                              id={`frame-card-${frameModelId}-${f.frame_index}`}
                              className={`frame-score-card frame-score-${level}`}
                              key={`${frameModelId}-${f.frame_index}-${f.timestamp_seconds}`}
                            >
                              <summary className="frame-score-summary">
                                <div className="frame-label">
                                  <strong>Frame {f.frame_index}</strong>
                                  <span>
                                    {Number(f.timestamp_seconds || 0).toFixed(
                                      2,
                                    )}{" "}
                                    s
                                  </span>
                                </div>

                                <div className="score-track">
                                  <div style={{ width: `${score * 100}%` }} />
                                </div>

                                <code>{score.toFixed(3)}</code>

                                <span className={`frame-score-tag ${level}`}>
                                  {level} score
                                </span>
                              </summary>

                              <div className="frame-score-detail">
                                <div className="frame-thumbnail-container">
                                  <img
                                    className="frame-thumbnail"
                                    src={`${API}/api/v1/analyses/${encodeURIComponent(analysis!.analysis_id)}/thumbnail?timestamp=${Number(f.timestamp_seconds || 0)}`}
                                    alt={`Video frame ${f.frame_index}`}
                                    loading="lazy"
                                    onClick={(event) => {
                                      const dialog = document.getElementById(
                                        "frame-preview-dialog",
                                      ) as HTMLDialogElement | null;

                                      const preview = document.getElementById(
                                        "frame-preview-image",
                                      ) as HTMLImageElement | null;

                                      if (dialog && preview) {
                                        preview.src = event.currentTarget.src;
                                        dialog.showModal();
                                      }
                                    }}
                                  />
                                  <span>Sampled frame preview</span>
                                </div>

                                <span>Frame index: {f.frame_index}</span>

                                <span>
                                  Timestamp:{" "}
                                  {Number(f.timestamp_seconds || 0).toFixed(2)}{" "}
                                  seconds
                                </span>

                                <span>
                                  Uncalibrated manipulated score:{" "}
                                  {score.toFixed(6)}
                                </span>

                                <p>
                                  This is an uncalibrated model score, not a
                                  probability or confirmation that this frame is
                                  manipulated.
                                </p>
                              </div>
                            </details>
                          );
                        })}
                      </div>
                    </section>
                  )}

                  {inferenceReady && fakestormerReady && (
                    <section className="panel fakestormer-temporal-panel">
                      <div className="panel-title">
                        <div>
                          <span className="eyebrow">
                            CHALLENGER 2 Â· FAKESTORMER
                          </span>
                          <h3>Temporal localization evidence</h3>
                          <p>
                            Four temporal samples from FakeSTormerâ€™s video-level
                            inference, displayed separately from the other
                            detectorsâ€™ frame-score timeline.
                          </p>
                        </div>
                        <span className="mini-tag">
                          {fakestormerTemporalValues.length} samples
                        </span>
                      </div>

                      {fakestormerTemporalValues.length > 0 ? (
                        <>
                          <div className="fakestormer-temporal-list">
                            {fakestormerTemporalValues.map(
                              (value: number, index: number) => {
                                const boundedValue = Math.max(
                                  0,
                                  Math.min(1, value),
                                );
                                const frameIndex =
                                  fakestormerSampleIndices[index];
                                const hasFrameIndex =
                                  Number.isFinite(frameIndex);
                                const timestamp =
                                  hasFrameIndex && fakestormerFps > 0
                                    ? frameIndex / fakestormerFps
                                    : null;

                                return (
                                  <div
                                    className="fakestormer-temporal-row"
                                    key={`fakestormer-temporal-${index}`}
                                  >
                                    <div className="fakestormer-temporal-meta">
                                      <strong>Sample {index + 1}</strong>
                                      <span>
                                        {hasFrameIndex
                                          ? `Frame ${frameIndex}`
                                          : "Frame index unavailable"}
                                        {timestamp !== null
                                          ? ` Â· ${timestamp.toFixed(2)} s`
                                          : ""}
                                      </span>
                                      <strong>{value.toFixed(4)}</strong>
                                    </div>
                                    <div
                                      className="fakestormer-temporal-track"
                                      role="img"
                                      aria-label={`Sample ${index + 1}: uncalibrated temporal localization value ${value.toFixed(4)}`}
                                    >
                                      <div
                                        className="fakestormer-temporal-fill"
                                        style={{
                                          width: `${boundedValue * 100}%`,
                                        }}
                                      />
                                    </div>
                                  </div>
                                );
                              },
                            )}
                          </div>

                          <div className="fakestormer-temporal-summary">
                            <div>
                              <span className="eyebrow">
                                FINAL TEMPORAL-LOCALIZATION VALUE
                              </span>
                              <strong>
                                {fakestormerFinalValue !== null
                                  ? fakestormerFinalValue.toFixed(4)
                                  : "Unavailable"}
                              </strong>
                            </div>
                            <div>
                              <span className="eyebrow">
                                OFFICIAL MODEL THRESHOLD
                              </span>
                              <strong>
                                {typeof fakestormer?.threshold === "number"
                                  ? fakestormer.threshold.toFixed(4)
                                  : "Unavailable"}
                              </strong>
                            </div>
                            <div>
                              <span className="eyebrow">
                                FAKESTORMER MODEL LABEL
                              </span>
                              <strong>
                                {fakestormer?.official_label || "Unavailable"}
                              </strong>
                            </div>
                          </div>
                        </>
                      ) : (
                        <p className="fakestormer-temporal-empty">
                          FakeSTormer completed, but no valid
                          temporal-localization values were returned for
                          display.
                        </p>
                      )}

                      <p className="fakestormer-temporal-disclaimer">
                        These are uncalibrated sigmoid temporal-localization
                        outputs, not probabilities. The official FakeSTormer
                        label follows its own test_vid threshold rule and is not
                        the VeriFrame verdict. VeriFrameâ€™s overall verdict
                        remains INCONCLUSIVE.
                      </p>
                    </section>
                  )}

                  {tab === "evidence" && (
                    <section className="panel">
                      <div className="panel-title">
                        <div>
                          <h3>Contradictory evidence & limitations</h3>

                          <p>Shown before supporting evidence by design.</p>
                        </div>
                      </div>

                      <div className="evidence-item warning">
                        <AlertTriangle size={17} />

                        <div>
                          <strong>
                            {inferenceReady
                              ? "Contradictory evidence & limitations"
                              : "Insufficient model evidence"}
                          </strong>

                          <p>
                            {inferenceReady
                              ? `${activeModelCount} detector(s) completed inference. GenD and Xception provide calibrated manipulated probabilities and jointly determine the verdict. Agreement produces REAL or FAKE; disagreement or an unavailable primary score produces INCONCLUSIVE. Effort provides supporting evidence, while FakeSTormer temporal-localization evidence is reported separately and is not treated as a calibrated probability.`
                              : result.committee?.reason ||
                                "No detector outputs are configured. The platform cannot support an authenticity claim."}
                          </p>
                        </div>
                      </div>

                      {(evidence?.model_evidence || []).map(
                        (x: any, i: number) => (
                          <div className="evidence-item" key={`model-${i}`}>
                            <div className="bullet" />

                            <div>
                              <strong>{x.label}</strong>

                              <p>
                                {x.value === undefined || x.value === null
                                  ? "Not available"
                                  : String(x.value)}

                                {x.detail ? ` â€” ${x.detail}` : ""}
                              </p>
                            </div>
                          </div>
                        ),
                      )}

                      {(evidence?.limitations || []).map(
                        (x: string, i: number) => (
                          <div className="evidence-item" key={`limit-${i}`}>
                            <div className="bullet" />

                            <div>
                              <strong>Limitation {i + 1}</strong>

                              <p>{x}</p>
                            </div>
                          </div>
                        ),
                      )}

                      <h3 className="subheading">Provenance evidence</h3>

                      {(evidence?.provenance_evidence || []).map(
                        (x: any, i: number) => (
                          <div className="metric-row" key={i}>
                            <span>{x.label}</span>

                            <strong>{String(x.value)}</strong>
                          </div>
                        ),
                      )}
                    </section>
                  )}

                  {tab === "stability" && (
                    <section className="panel stability-panel">
                      <div className="panel-title">
                        <div>
                          <h3>Stability Panel</h3>

                          <p>
                            Compare Xception scores under controlled video
                            transformations. Score-change labels are descriptive
                            heuristics, not statistical confidence levels.
                          </p>
                        </div>

                        <Activity size={18} />
                      </div>

                      <div className="stability-summary-grid">
                        <div className="stability-summary-card">
                          <span>Baseline score</span>

                          <strong>
                            {stability?.baseline_score != null
                              ? Number(stability.baseline_score).toFixed(6)
                              : "Not run"}
                          </strong>
                        </div>

                        <div className="stability-summary-card">
                          <span>Mean absolute score change</span>

                          <strong>
                            {stability?.score != null
                              ? Number(stability.score).toFixed(6)
                              : "â€”"}
                          </strong>
                        </div>

                        <div className="stability-summary-card">
                          <span>Decision flips</span>

                          <strong
                            className={
                              stability?.decision_flip_count > 0
                                ? "stability-value-warning"
                                : ""
                            }
                          >
                            {stability?.decision_flip_count ?? "â€”"}
                          </strong>
                        </div>
                      </div>

                      <button
                        className="primary-button stability-run-button"
                        disabled={
                          stabilityBusy || analysis.status !== "completed"
                        }
                        onClick={runStabilityCheck}
                      >
                        {stabilityBusy ? (
                          <>
                            <LoaderCircle className="spin" size={16} />
                            `` Running stability testsâ€¦
                          </>
                        ) : stability ? (
                          "Run Stability Check Again"
                        ) : (
                          "Run Stability Check"
                        )}
                      </button>

                      {analysis.status !== "completed" && (
                        <div className="stability-notice">
                          <AlertTriangle size={16} />

                          <span>
                            Wait for the video analysis to complete before
                            running stability tests.
                          </span>
                        </div>
                      )}

                      {stabilityError && (
                        <div
                          className="stability-alert stability-alert-error"
                          role="alert"
                        >
                          <AlertTriangle size={17} />

                          <div>
                            <strong>Stability check failed</strong>

                            <p>{stabilityError}</p>
                          </div>
                        </div>
                      )}

                      {stabilityBusy && (
                        <div className="stability-progress" role="status">
                          <LoaderCircle className="spin" size={17} />

                          <span>
                            Running baseline and transformed-video inference.
                            This may take a few minutes.
                          </span>
                        </div>
                      )}

                      {stability && (
                        <>
                          <div
                            className={`stability-alert ${
                              stability.status === "completed"
                                ? "stability-alert-success"
                                : stability.status === "partial"
                                  ? "stability-alert-warning"
                                  : "stability-alert-error"
                            }`}
                          >
                            {stability.status === "completed" ? (
                              <ShieldCheck size={18} />
                            ) : (
                              <AlertTriangle size={18} />
                            )}

                            <div>
                              <strong>
                                {stability.status === "completed"
                                  ? "All stability tests completed"
                                  : stability.status === "partial"
                                    ? "Stability check partially completed"
                                    : "Stability check failed"}
                              </strong>

                              <p>
                                Baseline output:{" "}
                                {stability.baseline_output || "Unavailable"}.{" "}
                                Completed{" "}
                                {stability.completed_test_count ??
                                  stabilityCompleted.length}{" "}
                                of{" "}
                                {stability.requested_test_count ??
                                  stabilityTests.length}{" "}
                                tests.
                              </p>

                              {stability.error && <p>{stability.error}</p>}
                            </div>
                          </div>

                          {largestStabilityChange && (
                            <div className="stability-largest-change">
                              <div>
                                <span>Largest observed score change</span>

                                <strong>{largestStabilityChange.label}</strong>
                              </div>

                              <strong>
                                {Number(
                                  largestStabilityChange.absolute_score_delta,
                                ).toFixed(6)}
                              </strong>
                            </div>
                          )}

                          {stabilityTests.length === 0 ? (
                            <div className="stability-empty-state">
                              <AlertTriangle size={20} />

                              <strong>
                                No perturbation results were returned
                              </strong>

                              <p>
                                Check the backend logs for the baseline
                                inference error, then try again.
                              </p>
                            </div>
                          ) : (
                            <div className="stability-test-list">
                              {stabilityTests.map((p: any) => {
                                const delta =
                                  p.absolute_score_delta == null
                                    ? null
                                    : Number(p.absolute_score_delta);

                                const severity =
                                  delta == null
                                    ? "unknown"
                                    : delta < 0.05
                                      ? "low"
                                      : delta < 0.15
                                        ? "moderate"
                                        : "high";

                                return (
                                  <div
                                    className="stability-test-card"
                                    key={p.id}
                                  >
                                    <div className="stability-test-heading">
                                      <div>
                                        <strong>
                                          {p.label || p.id || "Unnamed test"}
                                        </strong>

                                        <span className="stability-test-id">
                                          {p.id}
                                        </span>
                                      </div>

                                      <span
                                        className={`stability-badge ${
                                          p.status === "completed"
                                            ? "stability-badge-completed"
                                            : p.status === "skipped"
                                              ? "stability-badge-unknown"
                                              : "stability-badge-failed"
                                        }`}
                                      >
                                        {p.status === "completed"
                                          ? "Completed"
                                          : p.status === "skipped"
                                            ? "Skipped"
                                            : "Failed"}
                                      </span>
                                    </div>

                                    {p.status === "completed" &&
                                    delta != null ? (
                                      <>
                                        <div className="stability-test-metrics">
                                          <div>
                                            <span>Transformed score</span>

                                            <strong>
                                              {p.score != null
                                                ? Number(p.score).toFixed(6)
                                                : "Unavailable"}
                                            </strong>
                                          </div>

                                          <div>
                                            <span>Absolute score change</span>

                                            <strong
                                              className={`stability-delta-${severity}`}
                                            >
                                              {delta.toFixed(6)}
                                            </strong>
                                          </div>
                                        </div>

                                        <div
                                          className="stability-output-comparison"
                                          style={{
                                            display: "grid",

                                            gridTemplateColumns:
                                              "repeat(auto-fit, minmax(150px, 1fr))",

                                            gap: "10px",

                                            marginTop: "14px",

                                            padding: "12px",

                                            border:
                                              "1px solid rgba(148, 163, 184, 0.18)",

                                            borderRadius: "10px",

                                            background:
                                              "rgba(15, 23, 42, 0.22)",
                                          }}
                                        >
                                          <div>
                                            <span
                                              style={{
                                                display: "block",

                                                marginBottom: "5px",
                                              }}
                                            >
                                              Baseline output
                                            </span>

                                            <strong>
                                              {stability.baseline_output ||
                                                "Unavailable"}
                                            </strong>
                                          </div>

                                          <div>
                                            <span
                                              style={{
                                                display: "block",

                                                marginBottom: "5px",
                                              }}
                                            >
                                              Transformed output
                                            </span>

                                            <strong>
                                              {p.output || "Unavailable"}
                                            </strong>
                                          </div>
                                        </div>

                                        <div className="stability-test-footer">
                                          <span
                                            className={`stability-badge stability-severity-${severity}`}
                                          >
                                            {severity === "low"
                                              ? "Lower change"
                                              : severity === "moderate"
                                                ? "Moderate change"
                                                : "High change"}
                                          </span>

                                          <span
                                            className={`stability-badge ${
                                              p.decision_changed === true
                                                ? "stability-badge-warning"
                                                : p.decision_changed === false
                                                  ? "stability-badge-neutral"
                                                  : "stability-badge-unknown"
                                            }`}
                                          >
                                            {p.decision_changed === true
                                              ? "Decision changed"
                                              : p.decision_changed === false
                                                ? "Decision unchanged"
                                                : "Decision status unavailable"}
                                          </span>
                                        </div>
                                      </>
                                    ) : (
                                      <div className="stability-failure-detail">
                                        <strong>
                                          {p.status === "skipped"
                                            ? "Test skipped"
                                            : "Test failed"}
                                        </strong>

                                        <p>
                                          {p.reason ||
                                            (p.status === "skipped"
                                              ? "This test was not run. Check the backend response for the reason."
                                              : "The transformation or inference step did not return a usable score. Check the backend logs for details.")}
                                        </p>
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          )}

                          <p className="empty-inline">
                            {stability.score_definition ||
                              "Mean absolute difference between baseline and transformed-video scores. This is a diagnostic measure, not detection accuracy."}
                          </p>
                        </>
                      )}

                      {!stability && !stabilityBusy && !stabilityError && (
                        <div className="stability-empty-state">
                          <Activity size={22} />

                          <strong>Stability tests have not been run</strong>

                          <p>
                            Run the check to compare brightness, resize,
                            compression, blur, and noise transformations.
                          </p>
                        </div>
                      )}

                      <p className="empty-inline">
                        Score-change labels use simple display thresholds: below
                        0.05 is lower, 0.05 to below 0.15 is moderate, and 0.15
                        or above is high. These are not validated scientific
                        thresholds. Stability results do not prove authenticity
                        or detector accuracy. Interpret stability results
                        alongside the calibrated primary-model verdict and
                        supporting evidence.
                      </p>
                    </section>
                  )}

                  {tab === "provenance" && (
                    <section className="panel">
                      <div className="panel-title">
                        <div>
                          <h3>File provenance</h3>

                          <p>
                            File identity and metadata from the ingestion stage.
                          </p>
                        </div>

                        <Fingerprint size={18} />
                      </div>

                      <div className="metric-row">
                        <span>Original SHA-256</span>

                        <code>{analysis.sha256}</code>
                      </div>

                      <div className="metric-row">
                        <span>Container</span>

                        <strong>
                          {result.metadata?.container || "Unavailable"}
                        </strong>
                      </div>

                      <div className="metric-row">
                        <span>Video codec</span>

                        <strong>
                          {result.metadata?.video_codec || "Unavailable"}
                        </strong>
                      </div>

                      <div className="metric-row">
                        <span>Dimensions</span>

                        <strong>
                          {result.metadata?.width && result.metadata?.height
                            ? `${result.metadata.width} Ã— ${result.metadata.height}`
                            : "Unavailable"}
                        </strong>
                      </div>

                      <div className="metric-row">
                        <span>Duration</span>

                        <strong>
                          {result.metadata?.duration_seconds
                            ? `${result.metadata.duration_seconds.toFixed(2)} sec`
                            : "Unavailable"}
                        </strong>
                      </div>

                      <div className="metric-row">
                        <span>C2PA Content Credentials</span>

                        <strong>Not checked</strong>
                      </div>

                      <div className="metric-row">
                        <span>PDF evidence report</span>

                        <strong>Available for completed analyses</strong>
                      </div>
                    </section>
                  )}
                </>
              )}

              <div className="bottom-note">
                <ShieldCheck size={15} /> Results are evidence summaries, not
                proof. â€œLikely authenticâ€ must never be presented as verified
                authenticity.
              </div>
            </>
          )}

          <footer>
            VERIFRAME Â· EVIDENCE-FIRST VIDEO FORENSICS{" "}
            <span>Research prototype Â· v0.1.0</span>
          </footer>
        </div>
        <dialog id="frame-preview-dialog" className="frame-preview-dialog">
          <button
            className="frame-preview-close"
            onClick={() => {
              const dialog = document.getElementById(
                "frame-preview-dialog",
              ) as HTMLDialogElement | null;

              dialog?.close();
            }}
          >
            Close âœ•
          </button>

          <img id="frame-preview-image" alt="Enlarged forensic frame" />
        </dialog>
      </main>
    </div>
  );
}

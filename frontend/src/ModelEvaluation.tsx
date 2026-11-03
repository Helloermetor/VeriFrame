import { useCallback, useEffect, useState } from "react";

import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  LoaderCircle,
  Play,
  RefreshCw,
  Server,
  ShieldCheck,
} from "lucide-react";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

type Evaluation = {
  evaluation_id: string;

  status: string;

  created_at?: string;

  completed_at?: string;

  split?: string;

  batch_size?: number | null;

  threshold?: number;

  result?: any;

  error?: string;
};

type CurvePoint = {
  fpr?: number;

  tpr?: number;

  recall?: number;

  precision?: number;

  threshold: number | null;
};

type CurveLevel = {
  sample_count: number;

  positive_count: number;

  negative_count: number;

  roc_auc: number;

  average_precision: number;

  roc_curve: CurvePoint[];

  precision_recall_curve: CurvePoint[];
};

type CurveResponse = {
  evaluation_id: string;

  model: string;

  split: string;

  score_class: string;

  frame_level: CurveLevel;

  video_level: CurveLevel;

  interpretation_note?: string;
};

type ComparisonMetric = {
  first: number | null;

  second: number | null;

  delta: number | null;
};

type ComparisonResult = {
  first: {
    evaluation_id: string;

    created_at?: string;

    split?: string;

    checkpoint_sha256?: string;
  };

  second: {
    evaluation_id: string;

    created_at?: string;

    split?: string;

    checkpoint_sha256?: string;
  };

  same_split: boolean;

  same_checkpoint: boolean;

  frame_level: Record<string, ComparisonMetric>;

  video_level: Record<string, ComparisonMetric>;

  warnings: string[];

  delta_note: string;
};

type ComparisonCurves = {
  first: CurveResponse;

  second: CurveResponse;
};

function formatMetric(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Ã¢â‚¬â€";
  }

  return `${(value * 100).toFixed(2)}%`;
}

function formatDate(value?: string) {
  if (!value) return "Date unavailable";

  const date = new Date(value);

  return Number.isNaN(date.getTime())
    ? "Date unavailable"
    : date.toLocaleString();
}

function MetricCard({
  label,

  value,

  description,
}: {
  label: string;

  value: string;

  description: string;
}) {
  return (
    <div className="eval-metric-card">
      <span>{label}</span>

      <strong>{value}</strong>

      <small>{description}</small>
    </div>
  );
}

function CurveChart({
  title,

  subtitle,

  data,

  xKey,

  yKey,

  xLabel,

  yLabel,

  score,

  scoreLabel,

  showDiagonal = false,
}: {
  title: string;

  subtitle: string;

  data: CurvePoint[];

  xKey: "fpr" | "recall";

  yKey: "tpr" | "precision";

  xLabel: string;

  yLabel: string;

  score: number;

  scoreLabel: string;

  showDiagonal?: boolean;
}) {
  const chartData = data.filter(
    (point) =>
      typeof point[xKey] === "number" && typeof point[yKey] === "number",
  );

  return (
    <section className="panel eval-curve-card">
      <div className="panel-title eval-curve-title">
        <div>
          <h3>{title}</h3>

          <p>{subtitle}</p>
        </div>

        <div className="eval-curve-score">
          <strong>{formatMetric(score)}</strong>

          <small>{scoreLabel}</small>
        </div>
      </div>

      {chartData.length < 2 ? (
        <div className="empty-inline">Curve data is unavailable.</div>
      ) : (
        <div className="eval-curve-chart">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 12, right: 18, bottom: 24, left: 8 }}
            >
              <CartesianGrid
                stroke="rgba(148, 163, 184, 0.16)"
                strokeDasharray="3 3"
              />

              <XAxis
                type="number"
                dataKey={xKey}
                domain={[0, 1]}
                tick={{ fill: "#91a4c2", fontSize: 11 }}
                tickLine={false}
                axisLine={{ stroke: "#34445d" }}
                label={{
                  value: xLabel,

                  position: "insideBottom",

                  offset: -14,

                  fill: "#91a4c2",

                  fontSize: 11,
                }}
              />

              <YAxis
                type="number"
                dataKey={yKey}
                domain={[0, 1]}
                tick={{ fill: "#91a4c2", fontSize: 11 }}
                tickLine={false}
                axisLine={{ stroke: "#34445d" }}
                width={42}
                label={{
                  value: yLabel,

                  angle: -90,

                  position: "insideLeft",

                  fill: "#91a4c2",

                  fontSize: 11,
                }}
              />

              <Tooltip
                contentStyle={{
                  background: "#111c30",

                  border: "1px solid #34445d",

                  borderRadius: 10,

                  color: "#e6edf8",

                  fontSize: 12,
                }}
                labelFormatter={() => ""}
                formatter={(value) =>
                  typeof value === "number"
                    ? value.toFixed(4)
                    : String(value ?? "—")
                }
              />

              {showDiagonal && (
                <ReferenceLine
                  segment={[
                    { x: 0, y: 0 },

                    { x: 1, y: 1 },
                  ]}
                  stroke="#718096"
                  strokeDasharray="5 5"
                />
              )}

              <Line
                type="linear"
                dataKey={yKey}
                stroke="#69c8ed"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4, fill: "#9bdcf5" }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

function ComparisonCurveChart({
  title,

  firstData,

  secondData,

  xKey,

  yKey,

  xLabel,

  yLabel,
}: {
  title: string;

  firstData: CurvePoint[];

  secondData: CurvePoint[];

  xKey: "fpr" | "recall";

  yKey: "tpr" | "precision";

  xLabel: string;

  yLabel: string;
}) {
  const first = firstData.filter(
    (point) =>
      typeof point[xKey] === "number" && typeof point[yKey] === "number",
  );

  const second = secondData.filter(
    (point) =>
      typeof point[xKey] === "number" && typeof point[yKey] === "number",
  );

  return (
    <section className="panel eval-curve-card">
      <div className="panel-title">
        <div>
          <h3>{title}</h3>

          <p>Overlay of both saved evaluation runs</p>
        </div>
      </div>

      {first.length < 2 || second.length < 2 ? (
        <div className="empty-inline">
          Comparison curve data is unavailable.
        </div>
      ) : (
        <>
          <div className="eval-curve-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart margin={{ top: 12, right: 18, bottom: 24, left: 8 }}>
                <CartesianGrid
                  stroke="rgba(148, 163, 184, 0.16)"
                  strokeDasharray="3 3"
                />

                <XAxis
                  type="number"
                  dataKey={xKey}
                  domain={[0, 1]}
                  tick={{ fill: "#91a4c2", fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "#34445d" }}
                  label={{
                    value: xLabel,

                    position: "insideBottom",

                    offset: -14,

                    fill: "#91a4c2",

                    fontSize: 11,
                  }}
                />

                <YAxis
                  type="number"
                  dataKey={yKey}
                  domain={[0, 1]}
                  tick={{ fill: "#91a4c2", fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "#34445d" }}
                  width={42}
                  label={{
                    value: yLabel,

                    angle: -90,

                    position: "insideLeft",

                    fill: "#91a4c2",

                    fontSize: 11,
                  }}
                />

                <Tooltip
                  contentStyle={{
                    background: "#111c30",

                    border: "1px solid #34445d",

                    borderRadius: 10,

                    color: "#e6edf8",

                    fontSize: 12,
                  }}
                  formatter={(value) =>
                    typeof value === "number"
                      ? value.toFixed(4)
                      : String(value ?? "—")
                  }
                />

                {xKey === "fpr" && (
                  <ReferenceLine
                    segment={[
                      { x: 0, y: 0 },

                      { x: 1, y: 1 },
                    ]}
                    stroke="#718096"
                    strokeDasharray="5 5"
                  />
                )}

                <Line
                  name="First evaluation"
                  data={first}
                  dataKey={yKey}
                  type="linear"
                  stroke="#69c8ed"
                  strokeWidth={2.5}
                  dot={false}
                  isAnimationActive={false}
                />

                <Line
                  name="Second evaluation"
                  data={second}
                  dataKey={yKey}
                  type="linear"
                  stroke="#f2b66d"
                  strokeWidth={2.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="eval-curve-legend">
            <span>
              <i className="eval-legend-first" /> First evaluation
            </span>

            <span>
              <i className="eval-legend-second" /> Second evaluation
            </span>
          </div>
        </>
      )}
    </section>
  );
}

function ComparisonMetricTable({
  title,
  metrics,
}: {
  title: string;
  metrics: Record<string, ComparisonMetric>;
}) {
  const labels: Record<string, string> = {
    accuracy: "Accuracy",
    precision: "Precision",
    recall: "Recall",
    f1_score: "F1 score",
    roc_auc: "ROC-AUC",
    average_precision: "Average precision",
  };

  return (
    <section className="panel eval-comparison-metrics">
      <div className="panel-title">
        <div>
          <h3>{title}</h3>
          <p>Second evaluation minus first evaluation</p>
        </div>
      </div>

      <div className="eval-comparison-table-wrap">
        <table className="eval-comparison-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>First</th>
              <th>Second</th>
              <th>Difference</th>
              <th>Interpretation</th>
            </tr>
          </thead>

          <tbody>
            {Object.entries(metrics).map(([key, item]) => {
              const delta = item.delta;
              const hasDelta = delta != null;
              const isImproved = hasDelta && delta > 0;
              const isDeclined = hasDelta && delta < 0;

              const deltaClass = !hasDelta
                ? "eval-delta-neutral"
                : isImproved
                  ? "eval-delta-positive"
                  : isDeclined
                    ? "eval-delta-negative"
                    : "eval-delta-neutral";

              return (
                <tr key={key}>
                  <td>{labels[key] || key}</td>
                  <td>{formatMetric(item.first)}</td>
                  <td>{formatMetric(item.second)}</td>

                  <td>
                    <span className={deltaClass}>
                      {!hasDelta
                        ? "—"
                        : `${delta > 0 ? "+" : ""}${(delta * 100).toFixed(
                            2,
                          )} pp`}
                    </span>
                  </td>

                  <td>
                    <span className={`eval-change-label ${deltaClass}`}>
                      {!hasDelta
                        ? "Unavailable"
                        : delta > 0
                          ? "Higher"
                          : delta < 0
                            ? "Lower"
                            : "Unchanged"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function ModelEvaluation() {
  const [latest, setLatest] = useState<Evaluation | null>(null);

  const [history, setHistory] = useState<Evaluation[]>([]);
  const [historySearch, setHistorySearch] = useState("");
  const [historySplitFilter, setHistorySplitFilter] = useState("all");
  const [historyStatusFilter, setHistoryStatusFilter] = useState("all");

  const [loading, setLoading] = useState(true);

  const [running, setRunning] = useState(false);

  const [error, setError] = useState("");

  const [split, setSplit] = useState("test");

  const [batchSize, setBatchSize] = useState(32);

  const [threshold, setThreshold] = useState(0.5);

  const [curves, setCurves] = useState<CurveResponse | null>(null);

  const [curvesLoading, setCurvesLoading] = useState(false);

  const [curvesError, setCurvesError] = useState("");

  const [firstComparisonId, setFirstComparisonId] = useState("");

  const [secondComparisonId, setSecondComparisonId] = useState("");

  const [comparison, setComparison] = useState<ComparisonResult | null>(null);

  const [comparisonCurves, setComparisonCurves] =
    useState<ComparisonCurves | null>(null);

  const [comparisonLoading, setComparisonLoading] = useState(false);

  const [comparisonError, setComparisonError] = useState("");

  const [evaluationPdfLoading, setEvaluationPdfLoading] = useState(false);

  const [comparisonPdfLoading, setComparisonPdfLoading] = useState(false);

  const [pdfError, setPdfError] = useState("");

  const loadData = useCallback(async () => {
    try {
      setError("");

      const [latestResponse, historyResponse] = await Promise.all([
        fetch(`${API}/api/v1/model-evaluation`),

        fetch(`${API}/api/v1/model-evaluation/history`),
      ]);

      if (!latestResponse.ok || !historyResponse.ok) {
        throw new Error("Could not load model evaluation data.");
      }

      const latestData = await latestResponse.json();

      const historyData = await historyResponse.json();

      setLatest(latestData.status === "not_run" ? null : latestData);

      setHistory(
        Array.isArray(historyData.evaluations) ? historyData.evaluations : [],
      );
    } catch (err: any) {
      setError(err.message || "Unable to connect to the evaluation API.");
    } finally {
      setLoading(false);
    }
  }, []);
  const filteredHistory = history.filter((item) => {
    const matchesSearch = item.evaluation_id
      .toLowerCase()
      .includes(historySearch.trim().toLowerCase());

    const matchesSplit =
      historySplitFilter === "all" ||
      item.split?.toLowerCase() === historySplitFilter;

    const matchesStatus =
      historyStatusFilter === "all" ||
      item.status?.toLowerCase() === historyStatusFilter;

    return matchesSearch && matchesSplit && matchesStatus;
  });

  useEffect(() => {
    void loadData();
  }, [loadData]);
  //Poll the selected evaluation while it is running or queued.

  useEffect(() => {
    if (!latest?.evaluation_id) return;

    if (["completed", "failed", "interrupted"].includes(latest.status)) {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(
          `${API}/api/v1/model-evaluation/${latest.evaluation_id}`,
        );

        if (!response.ok) return;

        const data = await response.json();

        setLatest(data);

        if (["completed", "failed", "interrupted"].includes(data.status)) {
          void loadData();
        }
      } catch {
        //A temporary polling failure does not discard the current job.
      }
    }, 2500);

    return () => window.clearInterval(timer);
  }, [latest?.evaluation_id, latest?.status, loadData]);
  //Load ROC and Precision-Recall coordinates for the selected completed run.

  useEffect(() => {
    if (!latest?.evaluation_id || latest.status !== "completed") {
      setCurves(null);

      setCurvesError("");

      setCurvesLoading(false);

      return;
    }

    let cancelled = false;

    async function loadCurves() {
      setCurvesLoading(true);

      setCurvesError("");

      setCurves(null);

      try {
        const response = await fetch(
          `${API}/api/v1/model-evaluation/${latest!.evaluation_id}/curves`,
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            typeof data.detail === "string"
              ? data.detail
              : "Could not load ROC and Precision-Recall curves.",
          );
        }

        if (!cancelled) {
          setCurves(data as CurveResponse);
        }
      } catch (err: any) {
        if (!cancelled) {
          setCurvesError(
            err.message || "Unable to load evaluation curve data.",
          );
        }
      } finally {
        if (!cancelled) {
          setCurvesLoading(false);
        }
      }
    }

    void loadCurves();

    return () => {
      cancelled = true;
    };
  }, [latest?.evaluation_id, latest?.status]);

  useEffect(() => {
    const completed = history.filter((item) => item.status === "completed");

    if (!firstComparisonId && completed.length > 0) {
      setFirstComparisonId(completed[0].evaluation_id);
    }

    if (!secondComparisonId && completed.length > 1) {
      setSecondComparisonId(completed[1].evaluation_id);
    }
  }, [history, firstComparisonId, secondComparisonId]);

  async function runComparison() {
    if (!firstComparisonId || !secondComparisonId) {
      setComparisonError("Select two completed evaluations to compare.");

      return;
    }

    if (firstComparisonId === secondComparisonId) {
      setComparisonError("Choose two different evaluation runs.");

      return;
    }

    setComparisonLoading(true);

    setComparisonError("");

    setComparison(null);

    setComparisonCurves(null);

    try {
      const query = new URLSearchParams({
        first_id: firstComparisonId,

        second_id: secondComparisonId,
      });

      const [comparisonResponse, firstCurveResponse, secondCurveResponse] =
        await Promise.all([
          fetch(`${API}/api/v1/model-evaluation/compare?${query.toString()}`),

          fetch(`${API}/api/v1/model-evaluation/${firstComparisonId}/curves`),

          fetch(`${API}/api/v1/model-evaluation/${secondComparisonId}/curves`),
        ]);

      const comparisonData = await comparisonResponse.json();

      if (!comparisonResponse.ok) {
        throw new Error(
          typeof comparisonData.detail === "string"
            ? comparisonData.detail
            : "Could not compare the selected evaluations.",
        );
      }

      const firstCurveData = await firstCurveResponse.json();

      if (!firstCurveResponse.ok) {
        throw new Error(
          typeof firstCurveData.detail === "string"
            ? firstCurveData.detail
            : "Could not load the first evaluation curves.",
        );
      }

      const secondCurveData = await secondCurveResponse.json();

      if (!secondCurveResponse.ok) {
        throw new Error(
          typeof secondCurveData.detail === "string"
            ? secondCurveData.detail
            : "Could not load the second evaluation curves.",
        );
      }

      setComparison(comparisonData as ComparisonResult);

      setComparisonCurves({
        first: firstCurveData as CurveResponse,

        second: secondCurveData as CurveResponse,
      });
    } catch (err: any) {
      setComparisonError(
        err.message || "Unable to compare the selected evaluations.",
      );
    } finally {
      setComparisonLoading(false);
    }
  }

  async function downloadPdf(url: string, filename: string) {
    const response = await fetch(url);

    if (!response.ok) {
      let message = "Could not generate the PDF report.";

      try {
        const data = await response.json();
        message = data.detail || message;
      } catch {
        // Keep the default error message.
      }

      throw new Error(message);
    }

    const blob = await response.blob();
    const objectUrl = window.URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();

    window.setTimeout(() => window.URL.revokeObjectURL(objectUrl), 1000);
  }

  async function downloadEvaluationPDF() {
    if (!latest?.evaluation_id || latest.status !== "completed") return;

    setEvaluationPdfLoading(true);
    setPdfError("");

    try {
      const id = latest.evaluation_id;
      await downloadPdf(
        `${API}/api/v1/model-evaluation/${id}/report`,
        `veriframe_evaluation_${id.slice(0, 8)}.pdf`,
      );
    } catch (err: any) {
      setPdfError(err.message || "Evaluation PDF download failed.");
    } finally {
      setEvaluationPdfLoading(false);
    }
  }

  async function downloadComparisonPDF() {
    if (!comparison) return;

    setComparisonPdfLoading(true);
    setPdfError("");

    try {
      const params = new URLSearchParams({
        first_id: comparison.first.evaluation_id,
        second_id: comparison.second.evaluation_id,
      });

      await downloadPdf(
        `${API}/api/v1/model-evaluation/compare/report?${params.toString()}`,
        "veriframe_comparison_report.pdf",
      );
    } catch (err: any) {
      setPdfError(err.message || "Comparison PDF download failed.");
    } finally {
      setComparisonPdfLoading(false);
    }
  }

  async function runEvaluation() {
    if (running || ["queued", "running"].includes(latest?.status || "")) {
      return;
    }

    setRunning(true);

    setError("");

    try {
      const response = await fetch(`${API}/api/v1/model-evaluation/run`, {
        method: "POST",

        headers: { "Content-Type": "application/json" },

        body: JSON.stringify({
          split,

          batch_size: batchSize,

          threshold,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not start model evaluation.");
      }

      setLatest(data);

      await loadData();
    } catch (err: any) {
      setError(err.message || "Evaluation request failed.");
    } finally {
      setRunning(false);
    }
  }

  const summary = latest?.result?.summary;

  const frame = summary?.frame_metrics;

  const video = summary?.video_metrics_mean_score_per_frame_folder;

  const active = ["queued", "running"].includes(latest?.status || "");
  const evaluationSummary = [
    {
      label: "Evaluation status",
      value: latest?.status?.toUpperCase() || "NOT RUN",
    },
    {
      label: "Dataset split",
      value: latest?.split?.toUpperCase() || "N/A",
    },
    {
      label: "Evaluated frames",
      value: frame?.n?.toLocaleString() || "N/A",
    },
    {
      label: "Frame ROC-AUC",
      value: formatMetric(frame?.roc_auc),
    },
    {
      label: "Frame F1 score",
      value: formatMetric(frame?.f1_manipulated_class_1),
    },
    {
      label: "Completed at",
      value: formatDate(latest?.completed_at),
    },
  ];

  return (
    <div className="model-evaluation-page">
      <div className="eval-heading">
        <div>
          <div className="eyebrow">MODEL PERFORMANCE / VALIDATION</div>

          <h2>Model Evaluation Dashboard</h2>

          <p>
            Evaluate the custom Xception detector against the configured
            FaceForensics++ dataset split.
          </p>
        </div>

        <button
          className="secondary-button"
          onClick={() => {
            setLoading(true);

            void loadData();
          }}
          disabled={loading}
        >
          <RefreshCw size={15} />
          Refresh
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      {pdfError && <div className="error-box">{pdfError}</div>}

      <div className="eval-model-overview">
        <div className="eval-model-icon">
          <Activity size={25} />
        </div>

        <div className="eval-model-description">
          <strong>Custom DeepfakeBench Xception</strong>

          <span>Source-disjoint checkpoint Ã‚· Binary classification</span>

          <span>Class 0: Original Ã‚· Class 1: Manipulated</span>
        </div>

        <span className="status-pill neutral">
          {latest?.status?.toUpperCase() || "NOT RUN"}
        </span>
      </div>
      <section className="eval-provenance-panel">
        <div className="eval-section-heading">
          <h3>Model & Dataset Information</h3>
          <span>Evaluation configuration</span>
        </div>

        <div className="eval-provenance-grid">
          <div className="eval-provenance-item">
            <span>Model architecture</span>
            <strong>Xception</strong>
          </div>

          <div className="eval-provenance-item">
            <span>Model framework</span>
            <strong>DeepfakeBench</strong>
          </div>

          <div className="eval-provenance-item">
            <span>Dataset</span>
            <strong>FaceForensics++</strong>
          </div>

          <div className="eval-provenance-item">
            <span>Evaluation protocol</span>
            <strong>Source-disjoint split</strong>
          </div>

          <div className="eval-provenance-item">
            <span>Classification type</span>
            <strong>Binary classification</strong>
          </div>

          <div className="eval-provenance-item">
            <span>Input image size</span>
            <strong>256 × 256</strong>
          </div>
        </div>

        <div className="eval-class-mapping">
          <span>Class mapping</span>
          <div>
            <span className="eval-class-original">0 — Original</span>
            <span className="eval-class-manipulated">1 — Manipulated</span>
          </div>
        </div>
      </section>
      <section className="eval-summary-section">
        <div className="eval-section-heading">
          <h3>Evaluation at a glance</h3>
          <span>Latest evaluation overview</span>
        </div>

        <div className="eval-summary-grid">
          {evaluationSummary.map((item) => (
            <div className="eval-summary-card" key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="panel eval-run-panel">
        <div className="panel-title">
          <div>
            <h3>Run model evaluation</h3>

            <p>Configure the evaluation before starting a new run.</p>
          </div>

          <Play size={19} />
        </div>

        <div className="eval-controls">
          <label>
            Dataset split
            <select
              value={split}
              onChange={(e) => setSplit(e.target.value)}
              disabled={active || running}
            >
              <option value="test">Test</option>

              <option value="val">Validation</option>

              <option value="train">Train</option>
            </select>
          </label>

          <label>
            Batch size
            <select
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              disabled={active || running}
            >
              <option value={8}>8</option>

              <option value={16}>16</option>

              <option value={32}>32</option>

              <option value={64}>64</option>
            </select>
          </label>

          <label>
            Threshold
            <select
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              disabled={active || running}
            >
              <option value={0.5}>0.50</option>
            </select>
          </label>
        </div>

        <div className="eval-run-footer">
          <span>
            <AlertTriangle size={14} />
            Do not tune the threshold on the test split.
          </span>

          <button
            className="primary-button"
            onClick={runEvaluation}
            disabled={active || running || loading}
          >
            {active || running ? (
              <>
                <LoaderCircle className="spin" size={16} />
                Evaluation running...
              </>
            ) : (
              <>
                <Play size={15} />
                Run evaluation
              </>
            )}
          </button>
        </div>

        {active && (
          <div className="eval-progress">
            <LoaderCircle className="spin" size={16} />
            Evaluation is {latest?.status}. Keep the backend running.
          </div>
        )}
      </section>
      {error && !latest && !loading && (
        <div className="error-box eval-api-error">
          <AlertTriangle size={19} />

          <div>
            <strong>Unable to load evaluation data</strong>
            <p>{error}</p>

            <button
              className="secondary-button"
              onClick={() => {
                setLoading(true);
                void loadData();
              }}
            >
              Retry connection
            </button>
          </div>
        </div>
      )}
      {loading && !latest ? (
        <div className="panel empty-state">
          <LoaderCircle className="spin" size={23} />

          <strong>Loading evaluation data...</strong>
        </div>
      ) : !latest ? (
        <div className="panel empty-state">
          <Database size={25} />

          <strong>No model evaluation available</strong>

          <span>Run an evaluation to display model performance metrics.</span>
        </div>
      ) : (
        <>
          {latest.status === "completed" && summary && (
            <>
              <div className="eval-run-footer">
                <span>Export the selected evaluation and its metrics.</span>

                <button
                  className="secondary-button"
                  onClick={downloadEvaluationPDF}
                  disabled={evaluationPdfLoading}
                >
                  {evaluationPdfLoading ? (
                    <>
                      <LoaderCircle className="spin" size={16} />
                      Generating PDF...
                    </>
                  ) : (
                    "Download Evaluation PDF"
                  )}
                </button>
              </div>

              <div className="eval-section-heading">
                <h3>Frame-level performance</h3>

                <span>{frame?.n ?? "Ã¢â‚¬â€"} evaluated frames</span>
              </div>

              <div className="eval-metrics-grid">
                <MetricCard
                  label="Accuracy"
                  value={formatMetric(frame?.accuracy)}
                  description="Correct frame predictions"
                />

                <MetricCard
                  label="Precision"
                  value={formatMetric(frame?.precision_manipulated_class_1)}
                  description="Manipulated class"
                />

                <MetricCard
                  label="Recall"
                  value={formatMetric(frame?.recall_manipulated_class_1)}
                  description="Manipulated class"
                />

                <MetricCard
                  label="F1 Score"
                  value={formatMetric(frame?.f1_manipulated_class_1)}
                  description="Manipulated class"
                />

                <MetricCard
                  label="ROC-AUC"
                  value={formatMetric(frame?.roc_auc)}
                  description="Ranking performance"
                />

                <MetricCard
                  label="Average Precision"
                  value={formatMetric(frame?.average_precision)}
                  description="Precision-recall performance"
                />
              </div>

              <div className="eval-section-heading">
                <h3>Video-level performance</h3>

                <span>{video?.n ?? "Ã¢â‚¬â€"} frame-folder aggregates</span>
              </div>

              <div className="eval-metrics-grid">
                <MetricCard
                  label="Accuracy"
                  value={formatMetric(video?.accuracy)}
                  description="Aggregated predictions"
                />

                <MetricCard
                  label="Precision"
                  value={formatMetric(video?.precision_manipulated_class_1)}
                  description="Manipulated class"
                />

                <MetricCard
                  label="Recall"
                  value={formatMetric(video?.recall_manipulated_class_1)}
                  description="Manipulated class"
                />

                <MetricCard
                  label="F1 Score"
                  value={formatMetric(video?.f1_manipulated_class_1)}
                  description="Aggregated predictions"
                />

                <MetricCard
                  label="ROC-AUC"
                  value={formatMetric(video?.roc_auc)}
                  description="Ranking performance"
                />

                <MetricCard
                  label="Average Precision"
                  value={formatMetric(video?.average_precision)}
                  description="Precision-recall performance"
                />
              </div>

              <div className="grid-two eval-detail-grid">
                <section className="panel">
                  <div className="panel-title">
                    <div>
                      <h3>Frame confusion matrix</h3>

                      <p>Threshold: {frame?.threshold ?? latest.threshold}</p>
                    </div>
                  </div>

                  <ConfusionMatrix
                    matrix={
                      frame?.confusion_matrix_labels_0_original_1_manipulated
                    }
                  />
                </section>

                <section className="panel">
                  <div className="panel-title">
                    <div>
                      <h3>Video confusion matrix</h3>

                      <p>Mean class-1 score per frame folder</p>
                    </div>
                  </div>

                  <ConfusionMatrix
                    matrix={
                      video?.confusion_matrix_labels_0_original_1_manipulated
                    }
                  />
                </section>
              </div>

              <div className="eval-section-heading">
                <h3>ROC and Precision-Recall curves</h3>

                <span>Continuous class-1 manipulated scores</span>
              </div>

              {curvesLoading && (
                <div className="panel eval-curves-message">
                  <LoaderCircle className="spin" size={18} />
                  Loading evaluation curves...
                </div>
              )}

              {curvesError && (
                <div className="error-box">
                  Curve data could not be loaded: {curvesError}
                </div>
              )}

              {curves && (
                <>
                  <div className="eval-section-heading">
                    <h3>Frame-level curves</h3>

                    <span>
                      {curves.frame_level.sample_count} evaluated frames
                    </span>
                  </div>

                  <div className="grid-two eval-curves-grid">
                    <CurveChart
                      title="Frame ROC curve"
                      subtitle="True positive rate against false positive rate"
                      data={curves.frame_level.roc_curve}
                      xKey="fpr"
                      yKey="tpr"
                      xLabel="False positive rate"
                      yLabel="True positive rate"
                      score={curves.frame_level.roc_auc}
                      scoreLabel="ROC-AUC"
                      showDiagonal
                    />

                    <CurveChart
                      title="Frame Precision-Recall curve"
                      subtitle="Precision against recall"
                      data={curves.frame_level.precision_recall_curve}
                      xKey="recall"
                      yKey="precision"
                      xLabel="Recall"
                      yLabel="Precision"
                      score={curves.frame_level.average_precision}
                      scoreLabel="Average precision"
                    />
                  </div>

                  <div className="eval-section-heading">
                    <h3>Video-folder-level curves</h3>

                    <span>
                      {curves.video_level.sample_count} frame-folder aggregates
                    </span>
                  </div>

                  <div className="grid-two eval-curves-grid">
                    <CurveChart
                      title="Video-folder ROC curve"
                      subtitle="Mean frame score per folder"
                      data={curves.video_level.roc_curve}
                      xKey="fpr"
                      yKey="tpr"
                      xLabel="False positive rate"
                      yLabel="True positive rate"
                      score={curves.video_level.roc_auc}
                      scoreLabel="ROC-AUC"
                      showDiagonal
                    />

                    <CurveChart
                      title="Video-folder Precision-Recall curve"
                      subtitle="Mean frame score per folder"
                      data={curves.video_level.precision_recall_curve}
                      xKey="recall"
                      yKey="precision"
                      xLabel="Recall"
                      yLabel="Precision"
                      score={curves.video_level.average_precision}
                      scoreLabel="Average precision"
                    />
                  </div>

                  {curves.interpretation_note && (
                    <div className="eval-warning">
                      <AlertTriangle size={18} />

                      <div>
                        <strong>Curve interpretation</strong>

                        <p>{curves.interpretation_note}</p>
                      </div>
                    </div>
                  )}
                </>
              )}

              <section className="panel eval-provenance">
                <div className="panel-title">
                  <div>
                    <h3>Evaluation configuration</h3>

                    <p>Model and dataset provenance</p>
                  </div>

                  <ShieldCheck size={18} />
                </div>

                <InfoRow label="Model" value={summary.model} />

                <InfoRow
                  label="Dataset split"
                  value={summary.split_evaluated}
                />

                <InfoRow
                  label="Input image size"
                  value={`${summary.input_size} Ãƒ— ${summary.input_size}`}
                />

                <InfoRow label="Device" value={summary.device} />

                <InfoRow label="GPU" value={summary.gpu_name} />

                <InfoRow label="Threshold" value={String(summary.threshold)} />

                <InfoRow
                  label="Checkpoint SHA-256"
                  value={summary.checkpoint_sha256}
                />

                <InfoRow label="Checkpoint" value={summary.checkpoint} />

                <InfoRow label="Dataset CSV" value={summary.csv} />
              </section>

              <div className="eval-warning">
                <AlertTriangle size={18} />

                <div>
                  <strong>Research interpretation</strong>

                  <p>
                    Video-level metrics represent aggregation over frame
                    folders, not necessarily independently verified source
                    videos. The test split contains a limited number of
                    aggregates. These results do not establish generalization,
                    score calibration, or forensic certainty. The current model
                    score remains uncalibrated.
                  </p>
                </div>
              </div>
            </>
          )}

          {latest.status === "failed" && (
            <div className="error-box">
              Evaluation failed: {latest.error || "Check backend logs."}
            </div>
          )}

          {latest.status !== "completed" && latest.status !== "failed" && (
            <div className="panel empty-state">
              <Clock size={24} />

              <strong>Evaluation has not completed</strong>

              <span>
                Current status: {latest.status}. Metrics will appear when the
                backend finishes the evaluation.
              </span>
            </div>
          )}
        </>
      )}

      <section className="panel eval-comparison-panel">
        <div className="panel-title">
          <div>
            <h3>Evaluation comparison</h3>

            <p>
              Compare two completed runs using saved metrics and score curves.
            </p>
          </div>

          <Activity size={19} />
        </div>

        <div className="eval-comparison-controls">
          <label>
            First evaluation
            <select
              value={firstComparisonId}
              onChange={(event) => {
                setFirstComparisonId(event.target.value);

                setComparison(null);

                setComparisonCurves(null);
              }}
            >
              <option value="">Select first evaluation</option>

              {history

                .filter((item) => item.status === "completed")

                .map((item) => (
                  <option key={item.evaluation_id} value={item.evaluation_id}>
                    {item.split?.toUpperCase() || "RUN"} ·{" "}
                    {item.evaluation_id.slice(0, 8)} ·{" "}
                    {formatDate(item.created_at)}
                  </option>
                ))}
            </select>
          </label>

          <label>
            Second evaluation
            <select
              value={secondComparisonId}
              onChange={(event) => {
                setSecondComparisonId(event.target.value);

                setComparison(null);

                setComparisonCurves(null);
              }}
            >
              <option value="">Select second evaluation</option>

              {history

                .filter((item) => item.status === "completed")

                .map((item) => (
                  <option key={item.evaluation_id} value={item.evaluation_id}>
                    {item.split?.toUpperCase() || "RUN"} ·{" "}
                    {item.evaluation_id.slice(0, 8)} ·{" "}
                    {formatDate(item.created_at)}
                  </option>
                ))}
            </select>
          </label>

          <button
            className="primary-button"
            onClick={runComparison}
            disabled={
              comparisonLoading ||
              !firstComparisonId ||
              !secondComparisonId ||
              firstComparisonId === secondComparisonId
            }
          >
            {comparisonLoading ? (
              <>
                <LoaderCircle className="spin" size={16} />
                Comparing...
              </>
            ) : (
              <>
                <Activity size={16} />
                Compare evaluations
              </>
            )}
          </button>
        </div>

        {history.filter((item) => item.status === "completed").length < 2 && (
          <div className="empty-inline">
            At least two completed evaluation runs are required.
          </div>
        )}

        {comparisonError && <div className="error-box">{comparisonError}</div>}

        {comparison && (
          <div className="eval-comparison-results">
            <div className="eval-run-footer">
              <span>Export the metrics and comparison details.</span>

              <button
                className="secondary-button"
                onClick={downloadComparisonPDF}
                disabled={comparisonPdfLoading}
              >
                {comparisonPdfLoading ? (
                  <>
                    <LoaderCircle className="spin" size={16} />
                    Generating PDF...
                  </>
                ) : (
                  "Download Comparison PDF"
                )}
              </button>
            </div>

            <div className="eval-comparison-summary">
              <div>
                <span>First run</span>

                <strong>{comparison.first.evaluation_id.slice(0, 8)}</strong>

                <small>{comparison.first.split || "Split unavailable"}</small>
              </div>

              <div>
                <span>Second run</span>

                <strong>{comparison.second.evaluation_id.slice(0, 8)}</strong>

                <small>{comparison.second.split || "Split unavailable"}</small>
              </div>

              <div>
                <span>Dataset split</span>

                <strong>
                  {comparison.same_split ? "Matches" : "Different"}
                </strong>

                <small>
                  {comparison.same_split
                    ? "Like-for-like split"
                    : "Interpret differences cautiously"}
                </small>
              </div>

              <div>
                <span>Checkpoint</span>

                <strong>
                  {comparison.same_checkpoint ? "Same" : "Different"}
                </strong>

                <small>
                  {comparison.same_checkpoint
                    ? "Identical SHA-256"
                    : "Different SHA-256"}
                </small>
              </div>
            </div>

            {comparison.warnings.length > 0 && (
              <div className="eval-warning">
                <AlertTriangle size={18} />

                <div>
                  <strong>Comparison notes</strong>

                  {comparison.warnings.map((warning, index) => (
                    <p key={index}>{warning}</p>
                  ))}
                </div>
              </div>
            )}

            <ComparisonMetricTable
              title="Frame-level metric comparison"
              metrics={comparison.frame_level}
            />

            <ComparisonMetricTable
              title="Video-folder-level metric comparison"
              metrics={comparison.video_level}
            />

            {comparisonCurves && (
              <>
                <div className="eval-section-heading">
                  <h3>Overlaid frame-level curves</h3>

                  <span>First run vs second run</span>
                </div>

                <div className="grid-two eval-curves-grid">
                  <ComparisonCurveChart
                    title="Frame ROC comparison"
                    firstData={comparisonCurves.first.frame_level.roc_curve}
                    secondData={comparisonCurves.second.frame_level.roc_curve}
                    xKey="fpr"
                    yKey="tpr"
                    xLabel="False positive rate"
                    yLabel="True positive rate"
                  />

                  <ComparisonCurveChart
                    title="Frame Precision-Recall comparison"
                    firstData={
                      comparisonCurves.first.frame_level.precision_recall_curve
                    }
                    secondData={
                      comparisonCurves.second.frame_level.precision_recall_curve
                    }
                    xKey="recall"
                    yKey="precision"
                    xLabel="Recall"
                    yLabel="Precision"
                  />
                </div>

                <div className="eval-section-heading">
                  <h3>Overlaid video-folder-level curves</h3>

                  <span>Frame-folder aggregates</span>
                </div>

                <div className="grid-two eval-curves-grid">
                  <ComparisonCurveChart
                    title="Video-folder ROC comparison"
                    firstData={comparisonCurves.first.video_level.roc_curve}
                    secondData={comparisonCurves.second.video_level.roc_curve}
                    xKey="fpr"
                    yKey="tpr"
                    xLabel="False positive rate"
                    yLabel="True positive rate"
                  />

                  <ComparisonCurveChart
                    title="Video-folder Precision-Recall comparison"
                    firstData={
                      comparisonCurves.first.video_level.precision_recall_curve
                    }
                    secondData={
                      comparisonCurves.second.video_level.precision_recall_curve
                    }
                    xKey="recall"
                    yKey="precision"
                    xLabel="Recall"
                    yLabel="Precision"
                  />
                </div>
              </>
            )}
          </div>
        )}
      </section>

      <section className="panel eval-history">
        <div className="panel-title">
          <div>
            <h3>Evaluation history</h3>

            <p>Previously saved model evaluation runs.</p>
          </div>

          <Database size={18} />
        </div>
        <div className="eval-history-filters">
          <input
            type="text"
            placeholder="Search by evaluation ID..."
            value={historySearch}
            onChange={(e) => setHistorySearch(e.target.value)}
          />

          <select
            value={historySplitFilter}
            onChange={(e) => setHistorySplitFilter(e.target.value)}
          >
            <option value="all">All dataset splits</option>
            <option value="train">Train</option>
            <option value="val">Validation</option>
            <option value="test">Test</option>
          </select>

          <select
            value={historyStatusFilter}
            onChange={(e) => setHistoryStatusFilter(e.target.value)}
          >
            <option value="all">All statuses</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="running">Running</option>
            <option value="queued">Queued</option>
          </select>
        </div>
        {history.length === 0 ? (
          <div className="empty-line">No evaluation history available.</div>
        ) : filteredHistory.length === 0 ? (
          <div className="empty-line">
            No evaluations match your search and filters.
          </div>
        ) : (
          filteredHistory.map((item) => (
            <button
              className="eval-history-row"
              key={item.evaluation_id}
              onClick={async () => {
                try {
                  const response = await fetch(
                    `${API}/api/v1/model-evaluation/${item.evaluation_id}`,
                  );

                  if (!response.ok) {
                    throw new Error("Could not open this evaluation.");
                  }

                  setLatest(await response.json());

                  setError("");
                } catch (err: any) {
                  setError(err.message);
                }
              }}
            >
              <span className="eval-history-icon">
                {item.status === "completed" ? (
                  <CheckCircle2 size={17} />
                ) : (
                  <Server size={17} />
                )}
              </span>

              <span className="eval-history-info">
                <strong>
                  {item.split?.toUpperCase() || "EVALUATION"} Ã‚·{" "}
                  {item.evaluation_id.slice(0, 8)}
                </strong>

                <small>{formatDate(item.created_at)}</small>
              </span>

              <span className="status-pill neutral">{item.status}</span>
            </button>
          ))
        )}
      </section>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="metric-row">
      <span>{label}</span>

      <strong>{value == null ? "Unavailable" : String(value)}</strong>
    </div>
  );
}

function ConfusionMatrix({ matrix }: { matrix: any }) {
  if (!matrix) {
    return <div className="empty-inline">Confusion matrix unavailable.</div>;
  }

  return (
    <div className="eval-confusion">
      <div className="eval-confusion-labels">
        <span>Predicted: Original</span>

        <span>Predicted: Manipulated</span>
      </div>

      <div className="eval-confusion-grid">
        <div>
          <small>Actual original Ã‚· TN</small>

          <strong>{matrix.tn ?? "Ã¢â‚¬â€"}</strong>
        </div>

        <div>
          <small>Actual original Ã‚· FP</small>

          <strong>{matrix.fp ?? "Ã¢â‚¬â€"}</strong>
        </div>

        <div>
          <small>Actual manipulated Ã‚· FN</small>

          <strong>{matrix.fn ?? "Ã¢â‚¬â€"}</strong>
        </div>

        <div>
          <small>Actual manipulated Ã‚· TP</small>

          <strong>{matrix.tp ?? "Ã¢â‚¬â€"}</strong>
        </div>
      </div>
    </div>
  );
}

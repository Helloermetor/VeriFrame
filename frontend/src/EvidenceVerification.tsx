import { useState } from "react";
import {
  ShieldCheck,
  AlertTriangle,
  UploadCloud,
  LoaderCircle,
  FileCheck,
  FileWarning,
} from "lucide-react";

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

type VerificationResult = {
  status: "VERIFIED" | "TAMPERED" | "INCOMPLETE";
  verified: boolean;
  total_files: number;
  verified_files: number;
  missing_files: string[];
  modified_files: {
    filename: string;
    expected_sha256?: string;
    actual_sha256?: string;
    expected_size_bytes?: number;
    actual_size_bytes?: number;
  }[];
  errors: string[];
  analysis_id?: string;
  verdict?: string;
};

export default function EvidenceVerification() {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<VerificationResult | null>(null);

  async function verifyPackage() {
    if (!file) return;

    setBusy(true);
    setError("");
    setResult(null);

    try {
      const body = new FormData();
      body.append("package", file);

      const response = await fetch(`${API}/api/v1/evidence/verify`, {
        method: "POST",
        body,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Evidence verification failed.");
      }

      setResult(data);
    } catch (e: any) {
      setError(e.message || "Could not verify evidence package.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel verification-dashboard">
      <div className="panel-title">
        <div>
          <span className="eyebrow">FORENSIC INTEGRITY</span>
          <h3>Evidence package verification</h3>
          <p>
            Upload a VeriFrame evidence ZIP to check its files against the
            SHA-256 hashes recorded in its manifest.
          </p>
        </div>
        <ShieldCheck size={22} />
      </div>

      <div className="verification-upload">
        <UploadCloud size={30} />

        <strong>{file ? file.name : "Select an evidence package"}</strong>

        <span>
          {file
            ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
            : "Upload a VeriFrame ZIP package"}
        </span>

        <input
          type="file"
          accept=".zip,application/zip"
          onChange={(event) => {
            setFile(event.target.files?.[0] || null);
            setResult(null);
            setError("");
          }}
        />

        <button
          className="primary-button"
          disabled={!file || busy}
          onClick={verifyPackage}
        >
          {busy ? (
            <>
              <LoaderCircle className="spin" size={16} />
              Verifying package...
            </>
          ) : (
            <>
              <ShieldCheck size={16} />
              Verify evidence package
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="error-box" role="alert">
          {error}
        </div>
      )}

      {result && (
        <div className="verification-results">
          <div className={`verification-status ${result.status.toLowerCase()}`}>
            {result.verified ? (
              <ShieldCheck size={26} />
            ) : (
              <AlertTriangle size={26} />
            )}

            <div>
              <span>VERIFICATION RESULT</span>
              <h2>{result.status}</h2>
              <p>
                {result.status === "VERIFIED"
                  ? "All manifest-listed files match their recorded SHA-256 hashes and sizes."
                  : result.status === "TAMPERED"
                    ? "One or more files do not match the manifest."
                    : "The package is incomplete or could not be fully verified."}
              </p>
            </div>
          </div>

          <div className="verification-metrics">
            <div>
              <span>Total files</span>
              <strong>{result.total_files}</strong>
            </div>
            <div>
              <span>Verified files</span>
              <strong>{result.verified_files}</strong>
            </div>
            <div>
              <span>Missing files</span>
              <strong>{result.missing_files.length}</strong>
            </div>
            <div>
              <span>Modified files</span>
              <strong>{result.modified_files.length}</strong>
            </div>
          </div>

          <div className="verification-metadata">
            <div>
              <span>Analysis ID</span>
              <strong>{result.analysis_id || "Unavailable"}</strong>
            </div>
            <div>
              <span>Original assessment</span>
              <strong>{result.verdict || "Unavailable"}</strong>
            </div>
          </div>

          {result.missing_files.length > 0 && (
            <div className="verification-file-list">
              <h4>
                <FileWarning size={16} /> Missing files
              </h4>
              {result.missing_files.map((name) => (
                <div key={name}>{name}</div>
              ))}
            </div>
          )}

          {result.modified_files.length > 0 && (
            <div className="verification-file-list">
              <h4>
                <AlertTriangle size={16} /> Modified files
              </h4>
              {result.modified_files.map((item) => (
                <div key={item.filename}>
                  <strong>{item.filename}</strong>
                  <small>
                    Expected SHA-256: {item.expected_sha256 || "Unavailable"}
                  </small>
                  <small>
                    Actual SHA-256: {item.actual_sha256 || "Unavailable"}
                  </small>
                </div>
              ))}
            </div>
          )}

          {result.errors.length > 0 && (
            <div className="error-box">
              {result.errors.map((message, index) => (
                <p key={index}>{message}</p>
              ))}
            </div>
          )}

          {result.verified && (
            <div className="verification-success-note">
              <FileCheck size={17} />
              All listed package files passed integrity verification.
            </div>
          )}

          <p className="verification-disclaimer">
            A successful hash match confirms that the checked files match the
            supplied manifest. It does not authenticate the manifest creator or
            prove that the original video is authentic.
          </p>
        </div>
      )}
    </section>
  );
}

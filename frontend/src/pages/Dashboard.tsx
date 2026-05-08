import React, { useMemo, useRef, useState } from 'react';
import { CheckCircle2, FileUp, Loader2, Play, RotateCcw, TriangleAlert } from 'lucide-react';
import {
  continueAnalysis,
  parseCadFile,
  startPolling,
  submitClarification,
  type AnalysisResult,
  type CadParserPreview,
  type ReportType,
  type TaskStatus,
  type ThoughtType,
} from '../services/api-service';

type RunStatus = 'idle' | 'parsing' | 'parsed' | 'running' | 'clarification_needed' | 'complete' | 'failed';

export interface AnalysisMessage {
  id: string;
  sessionId: string;
  type: 'analysis' | 'followup';
  prompt: string;
  fileName: string | null;
  thoughts: ThoughtType[];
  progress: string | null;
  status: 'pending' | 'running' | 'clarification_needed' | 'complete' | 'failed';
  result: AnalysisResult | null;
  clarificationQuestion: string | null;
  error: string | null;
}

const featureOrder = [
  'source_format',
  'unit_assumption',
  'volume_mm3',
  'surface_area_mm2',
  'bbox_x_mm',
  'bbox_y_mm',
  'bbox_z_mm',
  'centroid_mm',
  'aspect_ratio',
  'dominant_axis',
  'min_wall_thickness_mm',
  'median_wall_thickness_mm',
  'has_thin_features',
  'thin_wall_threshold_mm',
  'triangle_count',
  'vertex_count',
  'is_watertight',
  'multi_body',
];

const formatLabel = (key: string) =>
  key.replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase());

const formatValue = (value: unknown): string => {
  if (value === null || value === undefined) return 'Not available';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4);
  if (Array.isArray(value)) return value.map(item => formatValue(item)).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

const getDefaultDescription = (filename?: string) =>
  filename?.replace(/\.[^.]+$/, '').replace(/[-_]+/g, ' ').trim() || 'uploaded STL part';

export const Dashboard: React.FC = () => {
  const [preview, setPreview] = useState<CadParserPreview | null>(null);
  const [status, setStatus] = useState<RunStatus>('idle');
  const [description, setDescription] = useState('');
  const [ecoWeight, setEcoWeight] = useState(0.7);
  const [progress, setProgress] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clarificationQuestion, setClarificationQuestion] = useState<string | null>(null);
  const [clarificationAnswer, setClarificationAnswer] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const cancelPollingRef = useRef<(() => void) | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const featureRows = useMemo(() => {
    if (!preview) return [];
    const keys = new Set([...featureOrder, ...Object.keys(preview.geometry)]);
    keys.delete('warnings');
    return Array.from(keys)
      .filter(key => Object.prototype.hasOwnProperty.call(preview.geometry, key))
      .map(key => ({ key, value: preview.geometry[key] }));
  }, [preview]);

  const parserWarnings = (preview?.geometry.warnings as string[] | undefined) ?? [];
  const isBusy = status === 'parsing' || status === 'running';

  const resetWorkflow = () => {
    cancelPollingRef.current?.();
    cancelPollingRef.current = null;
    setPreview(null);
    setStatus('idle');
    setDescription('');
    setEcoWeight(0.7);
    setProgress(null);
    setResult(null);
    setError(null);
    setClarificationQuestion(null);
    setClarificationAnswer('');
    setSessionId(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const applyTaskUpdate = (taskStatus: TaskStatus) => {
    setProgress(taskStatus.progress);
    setClarificationQuestion(taskStatus.clarification_question);
    setError(taskStatus.error);
  };

  const attachPolling = (taskId: string) => {
    cancelPollingRef.current?.();
    cancelPollingRef.current = startPolling(taskId, {
      onUpdate: applyTaskUpdate,
      onComplete: (taskStatus) => {
        applyTaskUpdate(taskStatus);
        setResult(taskStatus.result);
        setStatus('complete');
        cancelPollingRef.current = null;
      },
      onClarification: (question, taskStatus) => {
        applyTaskUpdate(taskStatus);
        setClarificationQuestion(question);
        setStatus('clarification_needed');
        cancelPollingRef.current = null;
      },
      onError: (message, taskStatus) => {
        applyTaskUpdate(taskStatus);
        setError(message);
        setStatus('failed');
        cancelPollingRef.current = null;
      },
    });
  };

  const handleFileSelected = async (file: File | undefined) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.stl')) {
      setError('Please upload an STL file.');
      setStatus('failed');
      return;
    }

    resetWorkflow();
    setStatus('parsing');
    setProgress('Running CAD parser...');

    try {
      const parsed = await parseCadFile(file);
      setPreview(parsed);
      setDescription(getDefaultDescription(parsed.filename));
      setStatus('parsed');
      setProgress(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'CAD parsing failed.');
      setStatus('failed');
      setProgress(null);
    }
  };

  const handleContinue = async () => {
    if (!preview || !description.trim()) return;
    setStatus('running');
    setProgress('Starting material analysis...');
    setError(null);
    setResult(null);

    try {
      const accepted = await continueAnalysis(preview.upload_id, description.trim(), ecoWeight);
      setSessionId(accepted.session_id);
      attachPolling(accepted.task_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to continue analysis.');
      setStatus('failed');
      setProgress(null);
    }
  };

  const handleClarification = async () => {
    if (!sessionId || !clarificationAnswer.trim()) return;
    setStatus('running');
    setProgress('Resuming material analysis...');
    setClarificationQuestion(null);

    try {
      const accepted = await submitClarification(sessionId, clarificationAnswer.trim());
      setClarificationAnswer('');
      attachPolling(accepted.task_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit clarification.');
      setStatus('failed');
      setProgress(null);
    }
  };

  const report = result?.report ?? null;

  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Lab Based Project</p>
            <h1 className="text-2xl font-semibold tracking-normal">Academic CAD Material Analysis Tool</h1>
          </div>
          <button
            className="inline-flex items-center gap-2 rounded-md border border-slate-300 px-3 py-2 text-sm font-medium hover:bg-slate-100"
            onClick={resetWorkflow}
          >
            <RotateCcw className="h-4 w-4" />
            Reset
          </button>
        </div>
      </header>

      <div className="mx-auto grid max-w-6xl gap-6 px-6 py-8">
        <section className="border border-slate-200 bg-white p-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-lg font-semibold">1. Upload STL File</h2>
              <p className="mt-1 text-sm text-slate-600">Upload one STL model to run the CAD parser preview.</p>
            </div>
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".stl"
                className="hidden"
                onChange={event => handleFileSelected(event.target.files?.[0])}
              />
              <button
                className="inline-flex items-center gap-2 rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={isBusy}
                onClick={() => fileInputRef.current?.click()}
              >
                {status === 'parsing' ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileUp className="h-4 w-4" />}
                Upload STL
              </button>
            </div>
          </div>
        </section>

        {preview && (
          <section className="border border-slate-200 bg-white">
            <div className="border-b border-slate-200 p-6">
              <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
                <div>
                  <h2 className="text-lg font-semibold">2. CAD Parser Output</h2>
                  <p className="mt-1 text-sm text-slate-600">{preview.filename}</p>
                </div>
                <span className="inline-flex w-fit items-center gap-2 rounded-md bg-emerald-50 px-3 py-1 text-sm font-medium text-emerald-700">
                  <CheckCircle2 className="h-4 w-4" />
                  Parser complete
                </span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-sm">
                <thead className="bg-slate-100 text-left text-xs uppercase tracking-wide text-slate-600">
                  <tr>
                    <th className="border-b border-slate-200 px-6 py-3 font-semibold">Feature</th>
                    <th className="border-b border-slate-200 px-6 py-3 font-semibold">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {featureRows.map(row => (
                    <tr key={row.key} className="odd:bg-white even:bg-slate-50">
                      <td className="border-b border-slate-100 px-6 py-3 font-medium text-slate-700">{formatLabel(row.key)}</td>
                      <td className="border-b border-slate-100 px-6 py-3 font-mono text-slate-900">{formatValue(row.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {parserWarnings.length > 0 && (
              <div className="border-t border-amber-200 bg-amber-50 px-6 py-4">
                <h3 className="flex items-center gap-2 text-sm font-semibold text-amber-900">
                  <TriangleAlert className="h-4 w-4" />
                  Parser Warnings
                </h3>
                <ul className="mt-2 grid gap-1 text-sm text-amber-800">
                  {parserWarnings.map((warning, index) => <li key={index}>{warning}</li>)}
                </ul>
              </div>
            )}
          </section>
        )}

        {preview && status !== 'complete' && (
          <section className="border border-slate-200 bg-white p-6">
            <h2 className="text-lg font-semibold">3. Continue Analysis</h2>
            <div className="mt-4 grid gap-4 md:grid-cols-[1fr_220px_auto] md:items-end">
              <label className="grid gap-1 text-sm font-medium text-slate-700">
                Part description
                <input
                  className="rounded-md border border-slate-300 px-3 py-2 text-sm font-normal text-slate-950 outline-none focus:border-slate-900"
                  value={description}
                  onChange={event => setDescription(event.target.value)}
                  disabled={isBusy}
                />
              </label>
              <label className="grid gap-1 text-sm font-medium text-slate-700">
                Recyclability priority: {ecoWeight.toFixed(2)}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={ecoWeight}
                  onChange={event => setEcoWeight(Number(event.target.value))}
                  disabled={isBusy}
                  className="h-10 accent-slate-900"
                />
              </label>
              <button
                className="inline-flex items-center justify-center gap-2 rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={isBusy || !description.trim() || status === 'clarification_needed'}
                onClick={handleContinue}
              >
                {status === 'running' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                Proceed
              </button>
            </div>
          </section>
        )}

        {(progress || status === 'running') && (
          <StatusPanel label={progress ?? 'Pipeline running...'} />
        )}

        {status === 'clarification_needed' && clarificationQuestion && (
          <section className="border border-amber-200 bg-amber-50 p-6">
            <h2 className="text-lg font-semibold text-amber-950">Clarification Required</h2>
            <p className="mt-2 text-sm text-amber-900">{clarificationQuestion}</p>
            <div className="mt-4 flex gap-3">
              <input
                className="flex-1 rounded-md border border-amber-300 px-3 py-2 text-sm outline-none focus:border-amber-700"
                value={clarificationAnswer}
                onChange={event => setClarificationAnswer(event.target.value)}
                placeholder="Enter clarification answer"
              />
              <button
                className="rounded-md bg-amber-900 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
                disabled={!clarificationAnswer.trim()}
                onClick={handleClarification}
              >
                Submit
              </button>
            </div>
          </section>
        )}

        {status === 'failed' && error && (
          <section className="border border-red-200 bg-red-50 p-6 text-sm text-red-800">
            <h2 className="text-lg font-semibold text-red-950">Analysis Failed</h2>
            <p className="mt-2">{error}</p>
          </section>
        )}

        {status === 'complete' && report && <ReportSection report={report} />}
      </div>
    </main>
  );
};

const StatusPanel: React.FC<{ label: string }> = ({ label }) => (
  <section className="flex items-center gap-3 border border-slate-200 bg-white p-4 text-sm text-slate-700">
    <Loader2 className="h-4 w-4 animate-spin text-slate-900" />
    {label}
  </section>
);

const ReportSection: React.FC<{ report: ReportType }> = ({ report }) => (
  <section className="border border-slate-200 bg-white">
    <div className="border-b border-slate-200 p-6">
      <h2 className="text-lg font-semibold">4. Material Recommendation</h2>
      {report.explanation?.text && (
        <p className="mt-3 whitespace-pre-line text-sm leading-6 text-slate-700">{report.explanation.text}</p>
      )}
    </div>

    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead className="bg-slate-100 text-left text-xs uppercase tracking-wide text-slate-600">
          <tr>
            <th className="border-b border-slate-200 px-6 py-3 font-semibold">Material</th>
            <th className="border-b border-slate-200 px-6 py-3 font-semibold">Eco Score</th>
            <th className="border-b border-slate-200 px-6 py-3 font-semibold">Confidence</th>
            <th className="border-b border-slate-200 px-6 py-3 font-semibold">Strength MPa</th>
            <th className="border-b border-slate-200 px-6 py-3 font-semibold">Stiffness GPa</th>
          </tr>
        </thead>
        <tbody>
          {[report.top_recommendation, ...report.alternatives].filter(Boolean).map(material => (
            <tr key={material!.material_name} className="odd:bg-white even:bg-slate-50">
              <td className="border-b border-slate-100 px-6 py-3 font-medium">{material!.material_name}</td>
              <td className="border-b border-slate-100 px-6 py-3">{material!.eco_score.toFixed(2)}</td>
              <td className="border-b border-slate-100 px-6 py-3">{material!.confidence.toFixed(2)}</td>
              <td className="border-b border-slate-100 px-6 py-3">{formatValue(material!.predicted_strength_MPa)}</td>
              <td className="border-b border-slate-100 px-6 py-3">{formatValue(material!.predicted_stiffness_GPa)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </section>
);

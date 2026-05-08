import React, { useMemo, useRef, useState } from 'react';
import { Activity, Box, ChevronDown, FileUp, Gauge, Loader2, Play, RotateCcw, Ruler, ShieldAlert } from 'lucide-react';
import {
  continueAnalysis,
  parseCadFile,
  previewFeatureTranslation,
  previewLoadAssumptions,
  startPolling,
  type AnalysisResult,
  type CadParserPreview,
  type FeaturePreview,
  type MaterialPrediction,
  type LoadPreview,
  type ReportType,
  type StructuralPrimitiveSummary,
  type TaskStatus,
  type ThoughtType,
} from '../services/api-service';

type RunStatus =
  | 'idle'
  | 'parsing'
  | 'parsed'
  | 'inferring_load'
  | 'load_review'
  | 'translating_features'
  | 'feature_review'
  | 'running'
  | 'complete'
  | 'failed';

export interface AnalysisMessage {
  id: string;
  sessionId: string;
  type: 'analysis' | 'followup';
  prompt: string;
  fileName: string | null;
  thoughts: ThoughtType[];
  progress: string | null;
  status: 'pending' | 'running' | 'complete' | 'failed';
  result: AnalysisResult | null;
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

const hiddenCadPreviewKeys = new Set([
  'warnings',
  'connected_components',
]);

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

const formatScore = (score: number | null | undefined): string =>
  typeof score === 'number' && Number.isFinite(score) ? score.toFixed(2) : 'Not available';

const getMaterialScore = (material: MaterialPrediction): number | undefined =>
  typeof material.eco_score === 'number' ? material.eco_score : material.confidence;

const getMaterialDescription = (material: MaterialPrediction): string => {
  if (material.description) return material.description;

  const strength = formatValue(material.predicted_strength_MPa);
  const stiffness = formatValue(material.predicted_stiffness_GPa);
  return `${material.material_name} is a candidate match for the translated requirements, with estimated strength ${strength} MPa and stiffness ${stiffness} GPa.`;
};

export const Dashboard: React.FC = () => {
  const [preview, setPreview] = useState<CadParserPreview | null>(null);
  const [status, setStatus] = useState<RunStatus>('idle');
  const [description, setDescription] = useState('');
  const [progress, setProgress] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loadPreview, setLoadPreview] = useState<LoadPreview | null>(null);
  const [featurePreview, setFeaturePreview] = useState<FeaturePreview | null>(null);
  const [expectedLoadN, setExpectedLoadN] = useState('');
  const [safetyFactor, setSafetyFactor] = useState('');
  const [error, setError] = useState<string | null>(null);
  const cancelPollingRef = useRef<(() => void) | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const featureRows = useMemo(() => {
    if (!preview) return [];
    const keys = new Set([...featureOrder, ...Object.keys(preview.geometry)]);
    hiddenCadPreviewKeys.forEach(key => keys.delete(key));
    return Array.from(keys)
      .filter(key => Object.prototype.hasOwnProperty.call(preview.geometry, key))
      .map(key => ({ key, value: preview.geometry[key] }));
  }, [preview]);

  const isBusy =
    status === 'parsing'
    || status === 'inferring_load'
    || status === 'translating_features'
    || status === 'running';
  const numericExpectedLoad = Number(expectedLoadN);
  const numericSafetyFactor = Number(safetyFactor);
  const canRunAnalysis =
    Boolean(loadPreview)
    && Number.isFinite(numericExpectedLoad)
    && numericExpectedLoad >= 0
    && Number.isFinite(numericSafetyFactor)
    && numericSafetyFactor >= 1;

  const resetWorkflow = () => {
    cancelPollingRef.current?.();
    cancelPollingRef.current = null;
    setPreview(null);
    setStatus('idle');
    setDescription('');
    setProgress(null);
    setResult(null);
    setLoadPreview(null);
    setFeaturePreview(null);
    setExpectedLoadN('');
    setSafetyFactor('');
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const applyTaskUpdate = (taskStatus: TaskStatus) => {
    setProgress(taskStatus.progress);
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

  const handleLoadPreview = async () => {
    if (!preview || !description.trim()) return;
    setStatus('inferring_load');
    setProgress('Inferring expected load and factor of safety...');
    setError(null);
    setLoadPreview(null);

    try {
      const inferred = await previewLoadAssumptions(preview.upload_id, description.trim());
      setLoadPreview(inferred);
      setFeaturePreview(null);
      setExpectedLoadN(String(inferred.expected_load_n));
      setSafetyFactor(String(inferred.safety_factor));
      setStatus('load_review');
      setProgress(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Load inference failed.');
      setStatus('failed');
      setProgress(null);
    }
  };

  const handleFeaturePreview = async () => {
    if (!preview || !description.trim() || !canRunAnalysis) return;
    setStatus('translating_features');
    setProgress('Translating CAD and load assumptions into material requirements...');
    setError(null);
    setFeaturePreview(null);

    try {
      const translated = await previewFeatureTranslation(
        preview.upload_id,
        description.trim(),
        numericExpectedLoad,
        numericSafetyFactor
      );
      setFeaturePreview(translated);
      setStatus('feature_review');
      setProgress(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Feature translation failed.');
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
      const accepted = await continueAnalysis(
        preview.upload_id,
        description.trim(),
        numericExpectedLoad,
        numericSafetyFactor
      );
      attachPolling(accepted.task_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to continue analysis.');
      setStatus('failed');
      setProgress(null);
    }
  };

  const report = result?.report ?? null;
  const showUpload = status === 'idle';
  const showCadTable = status === 'parsed' && preview;
  const showLoadEditor = status === 'load_review' && preview && loadPreview;
  const showFeatureReview = status === 'feature_review' && preview && featurePreview;
  const showStatus = status === 'parsing' || status === 'inferring_load' || status === 'translating_features' || status === 'running';

  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Lab Based Project</p>
            <h1 className="text-2xl font-semibold tracking-normal">Decision Sciences Tool for Sustainable Material Selection</h1>
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
        {showUpload && (
          <section className="grid min-h-[60vh] place-items-center border border-slate-200 bg-white p-6">
            <input
              ref={fileInputRef}
              type="file"
              accept=".stl"
              className="hidden"
              onChange={event => handleFileSelected(event.target.files?.[0])}
            />
            <button
              className="inline-flex items-center gap-2 rounded-md bg-slate-900 px-5 py-3 text-sm font-semibold text-white hover:bg-slate-800"
              onClick={() => fileInputRef.current?.click()}
            >
              <FileUp className="h-4 w-4" />
              Upload STL
            </button>
          </section>
        )}

        {showCadTable && (
          <section className="border border-slate-200 bg-white">
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
            <div className="flex justify-end border-t border-slate-200 p-4">
              <button
                className="inline-flex items-center justify-center gap-2 rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800"
                onClick={handleLoadPreview}
              >
                <Play className="h-4 w-4" />
                Next
              </button>
            </div>
          </section>
        )}

        {showLoadEditor && (
          <section className="border border-slate-200 bg-white p-6">
            <div className="grid gap-4 md:grid-cols-[1fr_1fr_auto] md:items-end">
              <label className="grid gap-1 text-sm font-medium text-slate-700">
                Expected load (N)
                <input
                  type="number"
                  min={0}
                  step={1}
                  className="rounded-md border border-slate-300 px-3 py-2 text-sm font-normal text-slate-950 outline-none focus:border-slate-900"
                  value={expectedLoadN}
                  onChange={event => setExpectedLoadN(event.target.value)}
                  disabled={isBusy}
                />
              </label>
              <label className="grid gap-1 text-sm font-medium text-slate-700">
                Factor of safety
                <input
                  type="number"
                  min={1}
                  step={0.1}
                  className="rounded-md border border-slate-300 px-3 py-2 text-sm font-normal text-slate-950 outline-none focus:border-slate-900"
                  value={safetyFactor}
                  onChange={event => setSafetyFactor(event.target.value)}
                  disabled={isBusy}
                />
              </label>
              <button
                className="inline-flex items-center justify-center gap-2 rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={isBusy || !canRunAnalysis}
                onClick={handleFeaturePreview}
              >
                {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                Next
              </button>
            </div>
          </section>
        )}

        {showFeatureReview && (
          <FeatureReview
            featurePreview={featurePreview}
            expectedLoadN={numericExpectedLoad}
            onRunAnalysis={handleContinue}
            isBusy={isBusy}
          />
        )}

        {showStatus && <StatusPanel />}

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

const StatusPanel: React.FC = () => (
  <section className="grid place-items-center border border-slate-200 bg-white p-8">
    <Loader2 className="h-4 w-4 animate-spin text-slate-900" />
  </section>
);

const getPrimitiveImpactScore = (primitive: StructuralPrimitiveSummary): number => {
  const pressureScore = primitive.applied_pressure_Pa > 0 ? primitive.applied_pressure_Pa / 1000 : 0;
  const bucklingScore = primitive.buckling_risk ? 1000 : 0;

  return Math.max(
    primitive.required_strength_MPa,
    primitive.stress_MPa,
    primitive.required_E_GPa * 100,
    primitive.applied_force_N / 10,
    pressureScore,
    bucklingScore
  );
};

const FeatureReview: React.FC<{
  featurePreview: FeaturePreview;
  expectedLoadN: number;
  onRunAnalysis: () => void;
  isBusy: boolean;
}> = ({ featurePreview, expectedLoadN, onRunAnalysis, isBusy }) => {
  const load = featurePreview.load_estimate;
  const vector = featurePreview.ml_input_vector;
  const primitives = vector.primitives ?? [];
  const impactPrimitives = [...primitives]
    .sort((left, right) => getPrimitiveImpactScore(right) - getPrimitiveImpactScore(left))
    .slice(0, 4);
  const governingPrimitive = vector.governing_primitive ?? primitives[0]?.archetype ?? 'unknown';

  const cards = [
    {
      title: 'Governing Primitive',
      value: formatLabel(governingPrimitive),
      icon: <Box className="h-4 w-4" />,
    },
    {
      title: 'Load Mode',
      value: formatLabel(load.primary_stress_mode),
      icon: <Activity className="h-4 w-4" />,
    },
    {
      title: 'Buckling Check',
      value: vector.buckling_risk ? 'Risk flagged' : 'No risk flagged',
      icon: <ShieldAlert className="h-4 w-4" />,
    },
    {
      title: 'Fatigue',
      value: vector.fatigue_critical ? 'Fatigue critical' : 'Not fatigue critical',
      icon: <Gauge className="h-4 w-4" />,
    },
  ];

  return (
    <section className="grid gap-6 border border-slate-200 bg-white p-6">
      <div className="grid gap-4 md:grid-cols-4">
        {cards.map(card => (
          <FeatureCard key={card.title} {...card} />
        ))}
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard label="Expected Load" value={`${formatValue(expectedLoadN)} N`} />
        <MetricCard
          label="Required Stiffness"
          value={`${formatValue(vector.required_stiffness_GPa)} GPa`}
        />
        <MetricCard
          label="Required Strength"
          value={`${formatValue(vector.required_tensile_strength_MPa)} MPa`}
        />
      </div>

      <div className="grid gap-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
          <Ruler className="h-4 w-4" />
          Structural Primitives
        </div>
        {impactPrimitives.length > 0 ? (
          <div className="grid gap-4 md:grid-cols-2">
            {impactPrimitives.map((primitive, index) => (
              <PrimitiveCard
                key={`${primitive.archetype}-${index}`}
                primitive={primitive}
                isGoverning={primitive.archetype === governingPrimitive}
              />
            ))}
          </div>
        ) : (
          <div className="border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
            Thermal-only case: no mechanical primitive analysis was required.
          </div>
        )}
      </div>

      <div className="flex justify-end">
        <button
          className="inline-flex items-center justify-center gap-2 rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
          disabled={isBusy}
          onClick={onRunAnalysis}
        >
          {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          Run Analysis
        </button>
      </div>
    </section>
  );
};

const FeatureCard: React.FC<{ title: string; value: string; icon: React.ReactNode }> = ({ title, value, icon }) => (
  <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-50">
    <div className="p-4">
      <p className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        {icon}
        {title}
      </p>
      <p className="mt-1 text-sm font-semibold text-slate-950">{value}</p>
    </div>
  </div>
);

const MetricCard: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="rounded-lg border border-slate-200 bg-white p-5">
    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
    <p className="mt-2 text-2xl font-semibold text-slate-950">{value}</p>
  </div>
);

const PrimitiveCard: React.FC<{ primitive: StructuralPrimitiveSummary; isGoverning: boolean }> = ({ primitive, isGoverning }) => (
  <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
    <div className="grid gap-4 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            {isGoverning ? 'Governing Primitive' : 'Detected Primitive'}
          </p>
          <h3 className="mt-1 text-base font-semibold text-slate-950">{formatLabel(primitive.archetype)}</h3>
        </div>
        {primitive.buckling_risk && (
          <span className="rounded-md border border-amber-300 bg-amber-100 px-2 py-1 text-xs font-semibold text-amber-900">
            Buckling
          </span>
        )}
      </div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <PrimitiveMetric label="Support" value={formatLabel(primitive.support_condition)} />
        <PrimitiveMetric label="Length" value={`${formatValue(primitive.effective_length_mm)} mm`} />
        <PrimitiveMetric label="Wall" value={`${formatValue(primitive.wall_thickness_mm)} mm`} />
        <PrimitiveMetric label="Force" value={`${formatValue(primitive.applied_force_N)} N`} />
        {primitive.applied_pressure_Pa > 0 && (
          <PrimitiveMetric label="Pressure" value={`${formatValue(primitive.applied_pressure_Pa)} Pa`} />
        )}
        <PrimitiveMetric label="Stress" value={`${formatValue(primitive.stress_MPa)} MPa`} />
        <PrimitiveMetric label="Strength Req." value={`${formatValue(primitive.required_strength_MPa)} MPa`} />
        <PrimitiveMetric label="Stiffness Req." value={`${formatValue(primitive.required_E_GPa)} GPa`} />
      </div>
    </div>
  </div>
);

const PrimitiveMetric: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="min-w-0">
    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
    <p className="mt-1 truncate font-mono text-slate-950">{value}</p>
  </div>
);

const ReportSection: React.FC<{ report: ReportType }> = ({ report }) => {
  const [expandedMaterial, setExpandedMaterial] = useState<string | null>(null);
  const recommendations = [report.top_recommendation, ...report.alternatives]
    .filter((material): material is MaterialPrediction => Boolean(material))
    .slice(0, 3);

  return (
    <section className="border border-slate-200 bg-white">
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead className="bg-slate-100 text-left text-xs uppercase tracking-wide text-slate-600">
            <tr>
              <th className="border-b border-slate-200 px-6 py-3 font-semibold">Rank</th>
              <th className="border-b border-slate-200 px-6 py-3 font-semibold">Material</th>
              <th className="border-b border-slate-200 px-6 py-3 font-semibold">Score</th>
              <th className="border-b border-slate-200 px-6 py-3 font-semibold">Details</th>
            </tr>
          </thead>
          <tbody>
            {recommendations.map((material, index) => {
              const isExpanded = expandedMaterial === material.material_name;

              return (
                <React.Fragment key={`${material.material_name}-${index}`}>
                  <tr className="odd:bg-white even:bg-slate-50">
                    <td className="border-b border-slate-100 px-6 py-3 font-mono text-slate-600">
                      {index + 1}
                    </td>
                    <td className="border-b border-slate-100 px-6 py-3 font-medium">{material.material_name}</td>
                    <td className="border-b border-slate-100 px-6 py-3">{formatScore(getMaterialScore(material))}</td>
                    <td className="border-b border-slate-100 px-6 py-3">
                      <button
                        className="inline-flex items-center gap-2 rounded-md border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100"
                        onClick={() => setExpandedMaterial(isExpanded ? null : material.material_name)}
                        aria-expanded={isExpanded}
                      >
                        <ChevronDown className={`h-4 w-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                        Description
                      </button>
                    </td>
                  </tr>
                  {isExpanded && (
                    <tr className="bg-slate-50">
                      <td className="border-b border-slate-100 px-6 py-4 text-sm leading-6 text-slate-700" colSpan={4}>
                        {getMaterialDescription(material)}
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
};

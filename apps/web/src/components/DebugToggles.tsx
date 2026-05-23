import { useStore } from "@/lib/store";

interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: () => void;
}

const Toggle = ({ label, checked, onChange }: ToggleProps) => (
  <label className="flex cursor-pointer items-center gap-1.5 rounded px-1.5 py-1 text-[11px] text-kivski-text hover:bg-kivski-panel-2">
    <input
      type="checkbox"
      checked={checked}
      onChange={onChange}
      className="h-3 w-3 accent-kivski-defender"
    />
    <span>{label}</span>
  </label>
);

const DebugToggles = () => {
  const showFov = useStore((s) => s.showFov);
  const showSound = useStore((s) => s.showSound);
  const showComms = useStore((s) => s.showComms);
  const showLastKnown = useStore((s) => s.showLastKnown);
  const showHeatmap = useStore((s) => s.showHeatmap);
  const toggleFov = useStore((s) => s.toggleFov);
  const toggleSound = useStore((s) => s.toggleSound);
  const toggleComms = useStore((s) => s.toggleComms);
  const toggleLastKnown = useStore((s) => s.toggleLastKnown);
  const toggleHeatmap = useStore((s) => s.toggleHeatmap);

  return (
    <div className="panel flex flex-col gap-0.5 p-1.5">
      <div className="px-1 pb-1 text-[9px] uppercase tracking-widest text-kivski-muted">
        Debug
      </div>
      <Toggle label="Show FoV" checked={showFov} onChange={toggleFov} />
      <Toggle label="Show Sound Radii" checked={showSound} onChange={toggleSound} />
      <Toggle label="Show Comms Arrows" checked={showComms} onChange={toggleComms} />
      <Toggle label="Show Last-Known" checked={showLastKnown} onChange={toggleLastKnown} />
      <Toggle label="Show Heatmap" checked={showHeatmap} onChange={toggleHeatmap} />
    </div>
  );
};

export default DebugToggles;

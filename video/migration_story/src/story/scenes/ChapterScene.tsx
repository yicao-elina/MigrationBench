import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { PROBE, PROBE_ORDER, ProbeKey, serifFamily, STAGE } from "../../theme";
import { ramp, travel } from "../anim";
import { DynamicsChart, ProfileChart, StaticChart } from "../charts";
import { Lattice } from "../Lattice";
import { PaperCard, StageScene } from "../Stage";
import { nebPath, slidingAsPath } from "../storyData";

const VERDICT_SEC = 4.6;
const SLIDE_AMPLITUDE_A = 2.1;

const CHAPTERS: Record<
  ProbeKey,
  { title: string; verdict: string; note: string }
> = {
  static: {
    title: "Is each frame right?",
    verdict: "At rest, the contenders look alike: 30–41 meV/atom.",
    note: "Foundation is not part of this held-out parity set.",
  },
  dynamics: {
    title: "Does it stay physical over 1 ns?",
    verdict:
      "In motion they split: two families leave physical diffusion and fail the rollout.",
    note: "Lattice motion is illustrative; D values are from the 1 ns MD campaign.",
  },
  within_gap: {
    title: "Can Cr slide along the gap?",
    verdict: "Every model under-predicts the in-gap peak: 0.61–0.98× DFT.",
    note: "Fixed DFT path, single-point energies; QE reference has an SCF warning.",
  },
  gap_to_ql: {
    title: "Can Cr reach the layer edge?",
    verdict:
      "No model sees an entry barrier (≤ 0.03 eV), but two miss the 0.1 eV downhill drive.",
    note: "Fixed DFT path, single-point energies; QE reference has an SCF warning.",
  },
  within_ql: {
    title: "Can Cr squeeze through the layer?",
    verdict: "Inside the layer, predicted peaks reach 1.7–13× the DFT value.",
    note: "Fixed DFT path, single-point energies; off-scale peaks labelled with true values.",
  },
  sliding: {
    title: "Can the layers slide past Cr?",
    verdict:
      "Under shear, no family finds the DFT barrier: profile RMSE 1.0–1.8 eV.",
    note: "Layer shift drawn schematically. Family-mean energies on the fixed path; DFT peak 2.57 eV at image 6, models peak at the endpoint.",
  },
};

const ChapterLattice: React.FC<{ probe: ProbeKey }> = ({ probe }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = travel(frame, fps, 0.8, 3.4);
  const size = { width: 640, height: 640 };
  switch (probe) {
    case "static":
      return (
        <Lattice
          pathKey="within_gap"
          t={0}
          frame={frame}
          {...size}
          showLayers
        />
      );
    case "dynamics":
      return (
        <Lattice
          pathKey="within_gap"
          t={0}
          frame={frame}
          {...size}
          jitter={0.22}
          showLayers
        />
      );
    case "sliding":
      return (
        <Lattice
          pathKey="within_gap"
          t={1}
          frame={frame}
          {...size}
          slideDx={Math.sin(t * Math.PI) * SLIDE_AMPLITUDE_A}
          showLayers
        />
      );
    default:
      return (
        <Lattice
          pathKey={probe}
          t={t}
          frame={frame}
          {...size}
          trail
          showLayers
        />
      );
  }
};

const ChapterChart: React.FC<{ probe: ProbeKey; letter: string }> = ({
  probe,
  letter,
}) => {
  switch (probe) {
    case "static":
      return <StaticChart startSec={1.0} letter={letter} />;
    case "dynamics":
      return <DynamicsChart startSec={1.0} letter={letter} />;
    case "sliding":
      return (
        <ProfileChart path={slidingAsPath()} startSec={0.8} letter={letter} />
      );
    default:
      return (
        <ProfileChart path={nebPath(probe)} startSec={0.8} letter={letter} />
      );
  }
};

const ProbeHeader: React.FC<{ probe: ProbeKey }> = ({ probe }) => {
  const meta = PROBE[probe];
  const index = PROBE_ORDER.indexOf(probe);
  return (
    <div
      style={{
        position: "absolute",
        left: 120,
        top: 64,
        right: 120,
        display: "flex",
        alignItems: "center",
        gap: 22,
        fontSize: 24,
        letterSpacing: 3,
        textTransform: "uppercase",
      }}
    >
      <div
        style={{
          width: 14,
          height: 14,
          borderRadius: 7,
          background: meta.color,
        }}
      />
      <div style={{ color: STAGE.ink, fontWeight: 600 }}>
        Probe {index + 1} · {meta.label}
      </div>
      <div style={{ color: STAGE.muted }}>{meta.asks}</div>
      <div style={{ flex: 1 }} />
      <div style={{ display: "flex", gap: 8 }}>
        {PROBE_ORDER.map((p, i) => (
          <div
            key={p}
            style={{
              width: 40,
              height: 4,
              background: i <= index ? PROBE[p].color : STAGE.faint,
              opacity: i <= index ? 1 : 0.5,
            }}
          />
        ))}
      </div>
    </div>
  );
};

export const ChapterScene: React.FC<{
  probe: ProbeKey;
  durationInFrames: number;
}> = ({ probe, durationInFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const meta = PROBE[probe];
  const chapter = CHAPTERS[probe];
  const letter = "abcdef"[PROBE_ORDER.indexOf(probe)];
  const card = ramp(frame, fps, 0.5, 0.9);

  return (
    <StageScene durationInFrames={durationInFrames}>
      <ProbeHeader probe={probe} />
      <div
        style={{
          position: "absolute",
          left: 120,
          top: 118,
          fontFamily: serifFamily,
          fontSize: 62,
          color: STAGE.ink,
          opacity: ramp(frame, fps, 0.2),
        }}
      >
        {chapter.title}
      </div>
      <div style={{ position: "absolute", left: 190, top: 250 }}>
        <ChapterLattice probe={probe} />
      </div>
      <PaperCard
        style={{
          position: "absolute",
          left: 940,
          top: 232,
          opacity: card,
          translate: `0px ${(1 - card) * 24}px`,
        }}
      >
        <ChapterChart probe={probe} letter={letter} />
      </PaperCard>
      <div
        style={{
          position: "absolute",
          left: 120,
          right: 120,
          bottom: 96,
          fontFamily: serifFamily,
          fontSize: 42,
          color: STAGE.ink,
          borderLeft: `4px solid ${meta.color}`,
          paddingLeft: 26,
          opacity: ramp(frame, fps, VERDICT_SEC),
        }}
      >
        {chapter.verdict}
      </div>
      <div
        style={{
          position: "absolute",
          left: 150,
          bottom: 50,
          fontSize: 22,
          color: STAGE.muted,
          opacity: ramp(frame, fps, VERDICT_SEC + 0.4),
        }}
      >
        {chapter.note}
      </div>
    </StageScene>
  );
};

import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { ATOM, PROBE, serifFamily, STAGE } from "../../theme";
import { clamp, ramp, travel } from "../anim";
import { Lattice } from "../Lattice";
import { StageScene } from "../Stage";

type SceneProps = { durationInFrames: number };

// Slow camera push-in applied to hero lattices (fraction of size per scene).
const PUSH_IN = 0.06;

export const Kicker: React.FC<{
  opacity: number;
  color?: string;
  children: React.ReactNode;
}> = ({ opacity, color = STAGE.muted, children }) => (
  <div
    style={{
      fontSize: 24,
      fontWeight: 600,
      letterSpacing: 6,
      textTransform: "uppercase",
      color,
      opacity,
    }}
  >
    {children}
  </div>
);

export const Headline: React.FC<{
  opacity: number;
  children: React.ReactNode;
  size?: number;
}> = ({ opacity, children, size = 84 }) => (
  <div
    style={{
      fontFamily: serifFamily,
      fontSize: size,
      fontWeight: 400,
      color: STAGE.ink,
      lineHeight: 1.08,
      marginTop: 18,
      opacity,
      translate: `0px ${(1 - opacity) * 24}px`,
    }}
  >
    {children}
  </div>
);

export const Body: React.FC<{
  opacity: number;
  children: React.ReactNode;
}> = ({ opacity, children }) => (
  <div
    style={{
      fontSize: 36,
      color: STAGE.muted,
      lineHeight: 1.4,
      marginTop: 28,
      opacity,
    }}
  >
    {children}
  </div>
);

const SpeciesKey: React.FC<{ opacity: number }> = ({ opacity }) => (
  <div
    style={{
      display: "flex",
      gap: 34,
      marginTop: 44,
      fontSize: 28,
      color: STAGE.muted,
      opacity,
    }}
  >
    {(["Te", "Sb", "Cr"] as const).map((sp) => (
      <span key={sp} style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span
          style={{
            width: 22,
            height: 22,
            borderRadius: 11,
            background: `radial-gradient(circle at 34% 30%, #fff 0%, ${ATOM[sp].color} 28%, ${ATOM[sp].color} 75%, ${STAGE.deep} 100%)`,
            display: "inline-block",
          }}
        />
        {sp}
      </span>
    ))}
  </div>
);

/** Cold open: the lattice assembles and Cr lights up in the vdW gap. */
export const ColdOpenScene: React.FC<SceneProps> = ({ durationInFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const host = ramp(frame, fps, 0.2, 2.0);
  const crIn = ramp(frame, fps, 2.0, 1.4);
  const push = interpolate(
    frame,
    [0, durationInFrames],
    [1, 1 + PUSH_IN],
    clamp,
  );

  return (
    <StageScene durationInFrames={durationInFrames}>
      <div
        style={{
          position: "absolute",
          left: 870,
          top: 150,
          opacity: host,
          scale: push * interpolate(host, [0, 1], [0.95, 1]),
        }}
      >
        <Lattice
          pathKey="within_gap"
          t={0}
          frame={frame}
          width={900}
          height={780}
          glow={crIn}
          hostOpacity={host}
          showLayers={crIn > 0.5}
        />
      </div>
      <div style={{ position: "absolute", left: 150, top: 330, width: 520 }}>
        <Kicker opacity={ramp(frame, fps, 2.6)} color={ATOM.Cr.color}>
          One atom, many scales
        </Kicker>
        <Headline opacity={ramp(frame, fps, 3.0)} size={132}>
          Meet <span style={{ color: ATOM.Cr.color }}>Cr</span>.
        </Headline>
        <Body opacity={ramp(frame, fps, 3.8)}>
          A single dopant atom in the van der Waals gap of Sb₂Te₃.
        </Body>
        <SpeciesKey opacity={ramp(frame, fps, 4.6)} />
      </div>
    </StageScene>
  );
};

/** The hero's journey: along the gap, then up into the quintuple layer (real NEB images). */
export const JourneyScene: React.FC<SceneProps> = ({ durationInFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const leg1 = travel(frame, fps, 0.6, 2.6);
  const leg2 = travel(frame, fps, 3.6, 2.8);
  const onLeg2 = frame >= 3.6 * fps;
  const push = interpolate(
    frame,
    [0, durationInFrames],
    [1 + PUSH_IN, 1],
    clamp,
  );

  return (
    <StageScene durationInFrames={durationInFrames}>
      <div style={{ position: "absolute", left: 870, top: 150, scale: push }}>
        <Lattice
          pathKey={onLeg2 ? "within_ql" : "within_gap"}
          t={onLeg2 ? leg2 : leg1}
          frame={frame}
          width={900}
          height={780}
          trail
          showLayers
        />
      </div>
      <div style={{ position: "absolute", left: 150, top: 300, width: 520 }}>
        <Kicker opacity={ramp(frame, fps, 0.2)}>The journey</Kicker>
        <Headline opacity={ramp(frame, fps, 0.3)}>Cr moves.</Headline>
        <Body opacity={ramp(frame, fps, 1.0)}>
          <span style={{ color: PROBE.within_gap.color, fontWeight: 600 }}>
            Along the gap
          </span>
          , then{" "}
          <span style={{ color: PROBE.within_ql.color, fontWeight: 600 }}>
            into a quintuple layer
          </span>
          .
        </Body>
        <Body opacity={ramp(frame, fps, 5.6)}>
          Every step here is a real NEB image of the 81-atom cell.
        </Body>
      </div>
    </StageScene>
  );
};

const GlassCard: React.FC<{
  opacity: number;
  children: React.ReactNode;
}> = ({ opacity, children }) => (
  <div
    style={{
      flex: 1,
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.12)",
      borderRadius: 6,
      padding: "44px 48px",
      opacity,
      translate: `0px ${(1 - opacity) * 20}px`,
    }}
  >
    {children}
  </div>
);

/** Why this is hard for MLIPs. */
export const ChallengeScene: React.FC<SceneProps> = ({ durationInFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const cards = [
    {
      title: "DFT",
      good: "Accurate energies and forces",
      bad: "Too costly for long, large trajectories",
    },
    {
      title: "MLIP",
      good: "Near-DFT speed-up, MD time scales",
      bad: "Learned mostly from near-equilibrium frames",
    },
  ];

  return (
    <StageScene durationInFrames={durationInFrames}>
      <div style={{ position: "absolute", left: 150, top: 110, right: 150 }}>
        <Kicker opacity={ramp(frame, fps, 0.1)}>The challenge</Kicker>
        <Headline opacity={ramp(frame, fps, 0.2)} size={76}>
          Accuracy or reach, rarely both
        </Headline>
        <div style={{ display: "flex", gap: 48, marginTop: 64 }}>
          {cards.map((c, i) => (
            <GlassCard key={c.title} opacity={ramp(frame, fps, 0.9 + i * 0.6)}>
              <div
                style={{
                  fontFamily: serifFamily,
                  fontSize: 58,
                  color: STAGE.ink,
                }}
              >
                {c.title}
              </div>
              <div style={{ fontSize: 34, marginTop: 26, color: STAGE.ink }}>
                + {c.good}
              </div>
              <div style={{ fontSize: 34, marginTop: 14, color: STAGE.muted }}>
                − {c.bad}
              </div>
            </GlassCard>
          ))}
        </div>
        <div
          style={{
            fontFamily: serifFamily,
            fontSize: 46,
            color: STAGE.ink,
            marginTop: 72,
            maxWidth: 1500,
            lineHeight: 1.3,
            opacity: ramp(frame, fps, 3.2),
          }}
        >
          Migration is decided by rare, high-energy moments — the configurations
          an MLIP has seen least.
        </div>
        <div
          style={{
            fontSize: 34,
            color: ATOM.Cr.color,
            marginTop: 28,
            opacity: ramp(frame, fps, 4.6),
          }}
        >
          So: can a model that looks accurate at rest follow Cr when it moves?
        </div>
      </div>
    </StageScene>
  );
};

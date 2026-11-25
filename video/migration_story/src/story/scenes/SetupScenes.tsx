import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { Marker } from "../../components/Marker";
import { ATOM, PROBE, PROBE_ORDER, SERIES_STYLE, STAGE } from "../../theme";
import { ramp, travel } from "../anim";
import { StageScene } from "../Stage";
import { Headline, Kicker } from "./OpeningScenes";

type SceneProps = { durationInFrames: number };

const CAST = [
  {
    key: "family_foundation",
    markers: ["family_foundation"],
    name: "Foundation",
    tag: "Zero-shot",
    line: "Broad materials prior (MACE-OMAT-0). Never sees our Cr AIMD.",
  },
  {
    key: "family_scratch600k",
    markers: ["family_scratch600k", "family_scratchmultit"],
    name: "From scratch",
    tag: "S600 · SMT",
    line: "Random init, trained only on target AIMD (600 K or multi-T).",
  },
  {
    key: "family_ft600k",
    markers: ["family_ft600k"],
    name: "Fine-tuned 600 K",
    tag: "FT600",
    line: "Foundation prior, adapted on 600 K target AIMD.",
  },
  {
    key: "family_ftmultit",
    markers: ["family_ftmultit"],
    name: "Fine-tuned multi-T",
    tag: "FTMT",
    line: "Foundation prior, adapted on multi-temperature AIMD.",
  },
];

/** The four model families. */
export const CastScene: React.FC<SceneProps> = ({ durationInFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <StageScene durationInFrames={durationInFrames}>
      <div style={{ position: "absolute", left: 150, right: 150, top: 110 }}>
        <Kicker opacity={ramp(frame, fps, 0.1)}>The contenders</Kicker>
        <Headline opacity={ramp(frame, fps, 0.2)} size={76}>
          Four ways to build a MACE potential
        </Headline>
        <div
          style={{
            fontSize: 32,
            color: STAGE.muted,
            marginTop: 18,
            opacity: ramp(frame, fps, 0.5),
          }}
        >
          for Cr-doped Sb₂Te₃. Colour is strategy; open marker, 600 K training;
          filled, multi-T.
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 40,
            marginTop: 56,
          }}
        >
          {CAST.map((c, i) => {
            const o = ramp(frame, fps, 1.0 + i * 0.5);
            return (
              <div
                key={c.key}
                style={{
                  borderTop: `3px solid ${SERIES_STYLE[c.key].color}`,
                  background: "rgba(255,255,255,0.04)",
                  borderRadius: 4,
                  padding: "34px 40px",
                  opacity: o,
                  translate: `${(1 - o) * 40}px 0px`,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
                  <svg width={c.markers.length * 34} height={30}>
                    {c.markers.map((m, mi) => (
                      <Marker
                        key={m}
                        shape={SERIES_STYLE[m].marker}
                        fill={SERIES_STYLE[m].fill}
                        color={SERIES_STYLE[m].color}
                        cx={15 + mi * 34}
                        cy={15}
                        size={24}
                        hollow={STAGE.mid}
                      />
                    ))}
                  </svg>
                  <span
                    style={{
                      fontSize: 46,
                      fontWeight: 600,
                      color: STAGE.ink,
                    }}
                  >
                    {c.name}
                  </span>
                  <span
                    style={{
                      fontSize: 26,
                      color: STAGE.muted,
                      fontWeight: 600,
                      letterSpacing: 2,
                    }}
                  >
                    {c.tag}
                  </span>
                </div>
                <div
                  style={{
                    fontSize: 31,
                    color: STAGE.muted,
                    marginTop: 14,
                    lineHeight: 1.3,
                  }}
                >
                  {c.line}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </StageScene>
  );
};

const CHIP_W = 250;
const CHIP_GAP = 22;

/** The diagnostic ladder: six probes, the hero hops across them. */
export const LadderScene: React.FC<SceneProps> = ({ durationInFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const hop = travel(frame, fps, 1.6, 3.6) * (PROBE_ORDER.length - 1);
  const crX = hop * (CHIP_W + CHIP_GAP) + CHIP_W / 2;
  const crLift = -Math.abs(Math.sin(hop * Math.PI)) * 60;

  return (
    <StageScene durationInFrames={durationInFrames}>
      <div style={{ position: "absolute", left: 150, right: 150, top: 110 }}>
        <Kicker opacity={ramp(frame, fps, 0.1)}>The diagnosis</Kicker>
        <Headline opacity={ramp(frame, fps, 0.2)} size={76}>
          Six probes, one hero
        </Headline>
        <div
          style={{
            fontSize: 32,
            color: STAGE.muted,
            marginTop: 18,
            opacity: ramp(frame, fps, 0.5),
          }}
        >
          Each probe asks the model to follow a different kind of configuration
          change.
        </div>
      </div>
      <div
        style={{
          position: "absolute",
          left: (1920 - (6 * CHIP_W + 5 * CHIP_GAP)) / 2,
          top: 470,
        }}
      >
        <div style={{ position: "relative", display: "flex", gap: CHIP_GAP }}>
          {PROBE_ORDER.map((p, i) => (
            <div
              key={p}
              style={{
                width: CHIP_W,
                opacity: ramp(frame, fps, 0.6 + i * 0.15),
                textAlign: "center",
              }}
            >
              <div
                style={{
                  height: 4,
                  background: PROBE[p].color,
                }}
              />
              <div
                style={{
                  fontSize: 34,
                  fontWeight: 600,
                  color: STAGE.ink,
                  marginTop: 22,
                }}
              >
                {PROBE[p].label}
              </div>
              <div style={{ fontSize: 26, color: STAGE.muted, marginTop: 10 }}>
                {PROBE[p].asks}
              </div>
            </div>
          ))}
          <div
            style={{
              position: "absolute",
              left: crX - 26,
              top: -90 + crLift,
              width: 52,
              height: 52,
              borderRadius: 26,
              background: `radial-gradient(circle at 34% 30%, #fff 0%, ${ATOM.Cr.color} 30%, ${ATOM.Cr.color} 75%, #7a5a1c 100%)`,
              boxShadow: `0 0 50px 8px ${ATOM.Cr.color}88`,
              opacity: ramp(frame, fps, 1.2),
            }}
          />
        </div>
        <div
          style={{
            marginTop: 70,
            display: "flex",
            alignItems: "center",
            gap: 20,
            fontSize: 26,
            letterSpacing: 3,
            textTransform: "uppercase",
            color: STAGE.muted,
            opacity: ramp(frame, fps, 2.4),
          }}
        >
          <span>pointwise</span>
          <div
            style={{
              flex: 1,
              height: 2,
              background: `linear-gradient(90deg, ${STAGE.faint}, ${STAGE.ink})`,
            }}
          />
          <span>▶ collective motion</span>
        </div>
      </div>
      <div
        style={{
          position: "absolute",
          left: 150,
          bottom: 70,
          fontSize: 24,
          color: STAGE.muted,
          opacity: ramp(frame, fps, 3.0),
        }}
      >
        Probe labels define tasks; they are not a calibrated scalar of
        distribution shift.
      </div>
    </StageScene>
  );
};

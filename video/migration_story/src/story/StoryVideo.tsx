import { AbsoluteFill, Composition, Series, useVideoConfig } from "remotion";
import { FPS, HEIGHT, PROBE_ORDER, WIDTH } from "../theme";
import { ChapterScene } from "./scenes/ChapterScene";
import { FinaleScene, ScorecardScene } from "./scenes/ClosingScenes";
import {
  ChallengeScene,
  ColdOpenScene,
  JourneyScene,
} from "./scenes/OpeningScenes";
import { CastScene, LadderScene } from "./scenes/SetupScenes";
import { Stage } from "./Stage";

const SECONDS = {
  open: 7,
  journey: 8,
  challenge: 8,
  cast: 7,
  ladder: 7,
  chapter: 8,
  scorecard: 9,
  finale: 8,
};
const TOTAL_SECONDS =
  SECONDS.open +
  SECONDS.journey +
  SECONDS.challenge +
  SECONDS.cast +
  SECONDS.ladder +
  SECONDS.chapter * PROBE_ORDER.length +
  SECONDS.scorecard +
  SECONDS.finale;

export const StoryVideo: React.FC = () => {
  const { fps } = useVideoConfig();
  return (
    <AbsoluteFill>
      <Stage />
      <Series>
        <Series.Sequence name="Cold open" durationInFrames={SECONDS.open * fps}>
          <ColdOpenScene durationInFrames={SECONDS.open * fps} />
        </Series.Sequence>
        <Series.Sequence
          name="Journey"
          durationInFrames={SECONDS.journey * fps}
        >
          <JourneyScene durationInFrames={SECONDS.journey * fps} />
        </Series.Sequence>
        <Series.Sequence
          name="Challenge"
          durationInFrames={SECONDS.challenge * fps}
        >
          <ChallengeScene durationInFrames={SECONDS.challenge * fps} />
        </Series.Sequence>
        <Series.Sequence name="Cast" durationInFrames={SECONDS.cast * fps}>
          <CastScene durationInFrames={SECONDS.cast * fps} />
        </Series.Sequence>
        <Series.Sequence
          name="Probe ladder"
          durationInFrames={SECONDS.ladder * fps}
        >
          <LadderScene durationInFrames={SECONDS.ladder * fps} />
        </Series.Sequence>
        {PROBE_ORDER.map((probe) => (
          <Series.Sequence
            key={probe}
            name={`Probe: ${probe}`}
            durationInFrames={SECONDS.chapter * fps}
          >
            <ChapterScene
              probe={probe}
              durationInFrames={SECONDS.chapter * fps}
            />
          </Series.Sequence>
        ))}
        <Series.Sequence
          name="Scorecard"
          durationInFrames={SECONDS.scorecard * fps}
        >
          <ScorecardScene durationInFrames={SECONDS.scorecard * fps} />
        </Series.Sequence>
        <Series.Sequence name="Finale" durationInFrames={SECONDS.finale * fps}>
          <FinaleScene durationInFrames={SECONDS.finale * fps} />
        </Series.Sequence>
      </Series>
    </AbsoluteFill>
  );
};

export const StoryComposition = () => (
  <Composition
    id="MigrationStory"
    component={StoryVideo}
    durationInFrames={TOTAL_SECONDS * FPS}
    fps={FPS}
    width={WIDTH}
    height={HEIGHT}
  />
);

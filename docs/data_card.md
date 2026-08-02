# Data card: Metrica Sports sample data

## Source and access

[github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data): three
anonymised matches of optical tracking with synchronised event annotation, published by Metrica
Sports as a sample of their commercial product.

Downloaded at setup time by `make data`, never committed. Each file's SHA-256 goes into
`data/raw/manifest.json`, and the dataset fingerprint is recorded in every model artifact and
evaluation report.

## Licence

**There is no formal licence.** The repository states only:

> Please be responsible with the use of this data.
> If you use it for anything public, please acknowledge the source.

That is permission to use, not a grant to redistribute. This project therefore:

- never vendors the data into the repository;
- downloads it locally at setup time;
- reproduces the attribution in the manifest, the README and here.

A reviewer who cannot or does not want to download it can still run the entire test suite and the
Slice 0 demo, which use a seeded synthetic match generated in-process.

## Contents

| Match | Format | Frames | Duration | Events |
|---|---|---|---|---|
| Sample_Game_1 | CSV (per-team tracking + events) | 145,006 | 96.7 min | 1,745 |
| Sample_Game_2 | CSV | 141,156 | 94.1 min | 1,935 |
| Sample_Game_3 | EPTS/FIFA XML + JSON events | 143,761 | 95.8 min | 3,620 |

~180 MB total.

## Coordinate conventions

Source coordinates are the unit square with `(0, 0)` at the **top left** and `(1, 1)` bottom right;
`(0.5, 0.5)` is the centre spot. Both pitches are 105 × 68 m.

Everything downstream uses canonical metres: origin at the centre spot, `+x` toward the right-hand
goal, `+y` toward the top touchline as drawn, so `x ∈ [-52.5, 52.5]`, `y ∈ [-34, 34]`.

Reorienting so the attacking team always attacks `+x` is a **180° rotation** (negate both axes), not
a mirror of `x`. Mirroring would swap the left and right wings and invert the handedness of every
angular feature.

## Sampling rate

25 Hz for all three matches, stated in the EPTS metadata and implied by the 0.04 s steps in the CSV
timestamps. No resampling is applied at ingestion; the model input is subsampled to 10 Hz.

## Anonymisation

Fully anonymised by the publisher: no player, team or competition names. Players appear as shirt
numbers (CSV) or opaque ids such as `P3578` (EPTS). Nothing in this project attempts to
de-anonymise them, and features are deliberately identity-invariant: players are ordered by
distance to ball or goal, never by shirt number, so a model cannot learn a specific individual.

## Playing direction

**The formats differ, and this shaped the whole orientation design.**

Sample game 3 declares `attack_direction_first_half` per team in its metadata, and labels each
player's `position_type` including which is the goalkeeper. Games 1 and 2 declare **neither**.
Direction also differs between the two CSV matches. The home team attacks `+x` in the first half
of game 1 and `−x` in the first half of game 2, so no convention can be assumed.

Direction is therefore inferred from a ranked hierarchy of evidence (metadata → pass progression →
goalkeeper position and team centroid → shot geometry), reconciled by weighted vote, and written
to `artifacts/reports/direction_<match>.json` with every signal and its margin.

Because game 3 declares the answer, it serves as ground truth for the inference used on the other
two. With the declaration withheld, the inferred direction matches it for all four team-periods,
asserted by `tests/unit/test_orientation.py::TestGameThreeGroundTruth`.

Two structural facts are enforced regardless of the vote: the two teams cannot attack the same end,
and a team must change ends between halves. Either violation stops preparation.

## Known quality characteristics

Measured, not assumed:

- **Ball position is absent in a large fraction of frames**, 39% in game 1. But 68% of those fall
  in dead-ball periods: the ball simply is not tracked while play is stopped. Restricted to in-play
  frames the rate is 18.0%, 24.6% and 27.2% for games 1, 2 and 3. Part of the remainder is genuine
  dropout and part is the dead-ball mask being approximate: it ends a stoppage at the next on-ball
  event, while the ball often is not tracked until slightly after.
- **Substitutes.** The CSV format carries fixed columns with `NaN` for players not on the pitch. The
  EPTS format writes only the eleven on the pitch, with column meaning changing at each of eleven
  `DataFormatSpecification` blocks; the parser scatters these into stable per-squad columns so a
  column means the same player all match.
- **Event taxonomy differs.** Game 3 includes `CARRY` events that games 1 and 2 do not. Nothing in
  the label or feature path may depend on `CARRY` being present.
- **Game 3 does not start at frame 1**, because there is pre-kickoff footage. Frames outside the declared
  period boundaries are discarded rather than assigned to a period.
- Goalkeepers are **declared** in game 3 and **inferred** in games 1 and 2 (as the outfield-extreme
  player). Which applies is recorded per player, so a report never implies the source said something
  it did not.

## Missing-data handling

Nothing is invented. Missing observations are never interpolated:

- Absent players remain `NaN` and are excluded from aggregates.
- A window with fewer than 80% valid frames, or whose final frame lacks the ball, is structurally
  invalid: it is never scored and can never produce an insight.
- File-level validation fails outright on unordered frames, backwards timestamps, implausible
  coordinates (>2% of samples more than 5 m off-pitch, which indicates a wrong coordinate
  convention rather than dropout), or events that do not align with the tracking frame range.

## Known biases and limitations

- **Three matches.** No basis for claims about any league, competition or season. Team identity,
  style, and even which two sides played are unknown, so no correction is possible.
- **One provider.** Coordinate conventions, smoothing and dropout behaviour are Metrica's. Another
  provider's feed would need re-validation.
- **Sample selection is unknown.** These matches were chosen by the publisher to demonstrate their
  product; they may be unrepresentatively clean.
- **Event annotation is human-derived** and its timing conventions are the publisher's. Possession
  inherits those conventions, including a lag at turnovers.
- The 5.80% positive rate is a property of these matches and this target definition, not a general
  base rate for football.

## Synthetic fixture

All automated tests run against `data/synthetic.py`, which generates a deterministic match from a
seed: possession sequences that progress upfield and sometimes enter the box, a defensive shape
that responds to the ball, and a consistent event stream. It can also be written out in Metrica's
exact CSV layout so the production parser is exercised against the real on-disk format.

It is football-shaped, not football: box entries occur roughly twice as often as in the real data,
which is deliberate, because it gives small fixtures enough positives to assert on. It is **not**
used for any reported metric.

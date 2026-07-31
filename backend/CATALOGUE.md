# What EEG data exists, and which of it is worth having

An index of every open EEG dataset on OpenNeuro, and a common form to read
them in. Nothing is downloaded by either step.

```bash
cd backend
.venv/bin/python -m tools.harvest_catalogue     # writes data/catalogue/eeg_catalogue.json
```

**447 EEG datasets. 32 TB. 25,742 subject-recordings.** For comparison, this
project currently holds four datasets and roughly 7 GB.

---

## Two decisions that shaped this

**It queries indices, not the open web.** OpenNeuro exposes a GraphQL API
carrying licences, subject counts and BIDS task metadata. Scraping search
results would return files with no provenance and no licence, which is worse
than useless for a project whose value rests on every number tracing back to a
control.

**It catalogues before it fetches.** Metadata for a thousand datasets costs a
few megabytes; the datasets are terabytes. What to download is a filtering
problem, and filtering wants an index.

## What "more data" actually means

Pooling heterogeneous EEG does not straightforwardly buy statistical power. A
motor-imagery recording contributes nothing to a question about emotion — the
two measure different constructs under different task demands, and stacking
them produces a larger table rather than a stronger test.

Two things do help:

- **More subjects on the same paradigm.** The emotion search ran 294
  correlations against 95 trials. That is underpowered whatever the analysis.
- **Independent datasets asking the same question**, so an effect can be tested
  for *replication* rather than re-fitted. This is the one that matters most
  here: every result in this project rests on a single dataset, so none of them
  has ever been replicated.

So entries are tagged by paradigm and the shortlist is grouped by the open
question each could answer.

---

## The index

| Paradigm | Datasets | Subjects | Why it matters here |
| --- | --- | --- | --- |
| affect | 32 | 1,545 | emotion decoding failed; needs independent replication |
| naturalistic | 27 | 3,697 | real scenes with ground truth — the standing gap |
| vision | 120 | 5,678 | appearance and spatial-limit work rests on one dataset |
| motor | 52 | 1,869 | strongest domain; more subjects sharpen the ceiling |
| memory | 57 | 4,564 | adjacent, untested here |
| rest | 71 | 7,678 | calibration baseline only |

*(Counts sum to more than 447 because the classifier is multi-label — a mixed
dataset appears under every question it could answer.)*

### Three classification bugs, and one that cannot be fixed

The first classifier took the first paradigm whose keyword appeared anywhere in
4,000 characters of README, with `affect` listed first. It filed a
polysomnography corpus under emotion. Replaced with field-weighted scoring: a
BIDS task label counts six, the dataset name four, a README mention one, capped
at three so a chatty paragraph cannot outvote a task label.

Second, `oddball` sat in the vision keyword list, so *auditory* oddball
datasets were tagged visual. Split into `visual oddball`, with `mmn` added to
audio.

Third, single-label classification hid the most useful corpus in the index. The
eleven Healthy Brain Network releases carry resting state, visual tasks *and*
movie watching; one label forced them to `rest`, so a shortlist built for the
naturalistic question never showed them. Now multi-label.

The one that cannot be fixed by any keyword list: **HBN's task labels are film
titles** — `DespicableMe`, `ThePresent`, `DiaryOfAWimpyKid`. The word "movie"
appears nowhere in the metadata. Keyword matching cannot recognise proper
nouns, so those eleven releases are tagged by hand in `OVERRIDES` after reading
their task lists. Expect other proper-noun datasets to be missed the same way;
the catalogue is a filter, not an oracle.

---

## Reading them all in one shape

`app/ingest/` wraps the three readers this project already has — BrainVision,
EEGLAB and EDF+ — behind one type. The readers are unchanged; nothing re-parses
bytes.

Two properties are worth naming.

**Samples stay lazy.** Two of the three readers memory-map their files because
recordings run to gigabytes. An eager `data` array would throw that away, so
`Recording` holds a window accessor and materialises only when asked.
Building an inventory across hundreds of files reads headers and nothing else.

**Sampling rates are not harmonised.** A 160 Hz recording and a 1000 Hz one
differ in what they can resolve, and a uniform rate would hide that — letting a
gamma-band result be computed from a recording that cannot represent gamma.
`usable_band()` reports the ceiling instead.

### The common electrode space

The real obstacle to combining EEG is not format, it is that the recordings do
not measure the same places. This project holds a 63-channel BrainVision cap, a
132-channel EGI net whose sensors are named `E1..E128`, and a 64-channel BCI2000
montage. Nothing lines up.

The honest common denominator is the **10-20 system**: nineteen positions
defined by fractions of head circumference rather than by hardware, present in
essentially every montage since 1958, and the set most consumer headsets sample
from. Verified against all three local datasets:

| Dataset | Format | Channels | Rate | Usable band | 10-20 coverage |
| --- | --- | --- | --- | --- | --- |
| ds004018 | BrainVision | 63 | 1000 Hz | ≤400 Hz | 18/19 |
| eegmmidb | EDF+ | 64 | 160 Hz | ≤64 Hz | **19/19** |
| ds003751 | EEGLAB | 132 | 250 Hz | ≤100 Hz | 18/19 |

The missing position in two of three is **Cz**, and that is correct rather than
a failure: Cz is the online reference in both, so it was never recorded as a
channel. It comes back as NaN.

Which is the other deliberate choice: **missing positions are never
interpolated.** A cap without T7 does not get a T7 invented from its
neighbours, because a decoder would then place weight on a channel that was
never recorded and the resulting number would be partly about the interpolation
kernel. Missing is reported as missing.

---

## What to fetch first

Nothing has been downloaded. The ranked candidates live in
`data/catalogue/eeg_catalogue.json`; the ones that would move a specific open
question:

- **Affect replication.** `ds005540` EmoEEG-MC (60 subjects, 51 GB) is
  explicitly built for cross-context emotion generalisation, which is the exact
  claim that failed here. `ds007172` EEG-Asymmetries (100 subjects, 12 GB)
  targets frontal asymmetry — the specific index this project weights lowest
  and would most like to test.
- **Naturalistic scenes.** The HBN releases, for film-clip viewing at a scale
  nothing else here approaches. Large: 170–260 GB per release.
- **Vision.** The PURSUE series (~290 subjects each, 8–16 GB) are standard ERP
  paradigms with big subject counts — cheap replication for the appearance
  results.

My recommendation is to start with **one** affect dataset rather than a broad
pull, and run the existing `emotion_correlate` pipeline on it as a *confirmatory*
test with the direction pre-specified. That is a far stronger design than the
exploratory search already run, and it costs 12 GB rather than a terabyte.

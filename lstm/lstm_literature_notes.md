# LSTM trajectory prediction: literature notes (Pallavi's part)

Status: I read the abstracts (and for Schöller et al. the abstract plus summary) of the papers below, not the full texts. Before citing a specific number or design detail in the report.

## What to say in the report (3 sentences)
LSTMs became the standard neural baseline for vehicle trajectory prediction around 2018, mostly on the NGSIM highway data: encode the recent motion with an LSTM, then decode future positions, sometimes with extra context about neighbours or the maneuver. Later work moved to multimodal outputs, because one past can lead to several futures. A constant-velocity model must still be reported, since it is surprisingly hard to beat on short horizons (Schöller et al.).

## Papers
| Paper | What it does | Why it matters for us |
|---|---|---|
| Deo & Trivedi, *Convolutional Social Pooling for Vehicle Trajectory Prediction*, arXiv 1805.06771 (2018) | LSTM encoder-decoder; a convolutional "social pooling" layer summarises neighbouring vehicles; outputs a distribution over maneuvers. Evaluated on NGSIM US-101 and I-80. | The classic LSTM design with interaction. Ours has no neighbour input (Waymo makes it possible; it is future work). |
| Deo & Trivedi, *Multi-Modal Trajectory Prediction of Surrounding Vehicles with Maneuver based LSTMs*, arXiv 1805.05499 (2018) | One LSTM output per maneuver class, giving several possible futures. NGSIM. | Motivates the multimodal (K=3) Transformer that Abdullah builds. |
| Altché & de La Fortelle, *An LSTM Network for Highway Trajectory Prediction*, arXiv 1801.07962 (2018) | LSTM predicting longitudinal and lateral motion on NGSIM US-101. | Simple single-path LSTM, closest to our model. |
| Park et al., *Sequence-to-Sequence Prediction of Vehicle Trajectory via LSTM Encoder-Decoder Architecture*, arXiv 1802.06338 (2018) | Encoder-decoder LSTM with beam search over an occupancy grid. | Shows the encoder-decoder alternative to our one-shot MLP head. |
| Schöller et al., *What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction*, IEEE RA-L 2020 | A constant-velocity model matches or beats state-of-the-art neural predictors on common pedestrian benchmarks. | Why the baseline is not a formality. Pedestrians, not vehicles, so do not over-claim. |
| Ettinger et al., *Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset*, arXiv 2104.10133 (2021) | The dataset paper: 10 Hz tracks, 1 s history plus 8 s future in the file, map data, ADE/FDE/minADE-style metrics. | Our data source. Cite it for dataset facts. |

## Our design choices versus the literature
- **One-shot output of all 30 positions** instead of step-by-step decoding. Chosen for simplicity and no error build-up. Try a step-by-step decoder in the tuning phase if time allows.
- **Loss = ADE itself** (mean Euclidean error), so we optimise what we report.
- **Residual option**: the LSTM predicts a correction to constant velocity (`--residual`). This is our own design choice for the tuning phase, not taken from a paper.
- **No map or neighbours.** Waymo has both. We leave them out so LSTM, CV and the single-path Transformer see the same inputs. Say this limitation in the report.

## Sources
- https://arxiv.org/abs/1805.06771
- https://arxiv.org/abs/1805.05499
- https://arxiv.org/abs/1801.07962
- https://arxiv.org/abs/1802.06338
- https://arxiv.org/abs/2104.10133
- Schöller et al., IEEE Robotics and Automation Letters, 2020 (search the title; confirm the volume and pages from IEEE or arXiv before citing)

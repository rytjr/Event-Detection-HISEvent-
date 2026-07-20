# Event Detection using HISEvent

This repository provides the implementation and experimental results for detecting political events from YouTube comments using the **HISEvent** framework. The project identifies political events from large-scale social media interactions and analyzes the political orientation of event participants by estimating user-level political scores from comment embeddings.

The proposed framework consists of three main stages:

- **Political Orientation Estimation:** Estimate user political scores from comment embeddings using ideology anchor representations.
- **Political Event Detection:** Detect political events from user interaction graphs with the HISEvent algorithm.
- **Event Participation Analysis:** Compare the political score distributions of event participants with that of the overall population using statistical tests and effect size analysis.

The repository contains the implementation of the proposed framework together with representative visualization results of detected political events.

## Event Distribution Examples

The figures below compare the political score distribution of participants in representative political events with that of the overall user population. The gray histogram represents the overall population, while the colored histogram represents participants in a specific event. Both distributions are peak-normalized to facilitate comparison of their shapes rather than their absolute frequencies. Vertical dashed lines indicate the mean political scores of the overall population and the event participants.

<table>
<tr>
<td align="center">
<img src="images/event13.png" width="260"><br>
<b>Event 1</b>
</td>
<td align="center">
<img src="images/event8.png" width="260"><br>
<b>Event 2</b>
</td>
<td align="center">
<img src="images/event24.png" width="260"><br>
<b>Event 3</b>
</td>
<td align="center">
<img src="images/event25.png" width="260"><br>
<b>Event 4</b>
</td>
</tr>

<tr>
<td align="center">
<img src="images/event1.png" width="260"><br>
<b>Event 5</b>
</td>
<td align="center">
<img src="images/event9.png" width="260"><br>
<b>Event 6</b>
</td>
<td align="center">
<img src="images/event20.png" width="260"><br>
<b>Event 7</b>
</td>
<td align="center">
<img src="images/event21.png" width="260"><br>
<b>Event 8</b>
</td>
</tr>
</table>

The event histograms are color-coded according to their dominant political orientation: **blue** indicates events with relatively progressive-leaning participants, while **red** indicates events with relatively conservative-leaning participants. The title of each figure reports Cohen's *d* and the number of matched participants used in the analysis.

**Additional event visualizations are available in the [`images/`](images/) directory, which contains the complete set of event distribution figures generated in this study.**
# Traffic Flow & Speed Prediction GUI

**Inference-only Streamlit application for interactive traffic flow and speed visualization.** Adjust link capacities in real time and observe predicted network effects using a pre-trained Feedforward Neural Network model.

## Overview

This GUI enables interactive exploration of how changes in link capacity affect predicted traffic flow and speed across a traffic network. It uses a pre-trained Feedforward Neural Network (FF) model trained on network-wide capacity context to provide fast, realistic predictions that account for upstream and downstream traffic interactions.

### Key Capabilities

- 🗺️ **Interactive network map** with real-time flow/speed coloring and directional arrows
- 🎨 **Multiple viewing modes** for flow and speed (absolute vs. relative metrics)
- ✏️ **Editable capacities** with instant prediction updates for what-if analysis
- 📊 **Time-series analysis** showing hourly patterns for selected links
- 🧭 **Local sensitivity analysis** revealing upstream/downstream network effects
- 📈 **Flow vs. Speed scatter** showing congestion relationships (when available)
- 📋 **Interactive data table** with sorting, filtering, and capacity editing
- 🔍 **Direct map interaction** - click links on the map to select them
- 💾 **Scenario-based analysis** with predefined or custom capacity scenarios

## Scope

- **Model Type:** Fixed Feedforward Neural Network (FF) - inference only
- **Training Model:** Trained on v3 global-capacity flow+speed data with network-wide capacity context
- **Input:** Link capacities (veh/hr) and hour of day (5–22)
- **Output:** Predicted vehicle flow (veh/hr) and speed (m/s) for each link
- **No Training:** This GUI performs inference only. Model training is documented separately in `README_MODEL_TRAINING.md`
- **Pre-trained Artifacts:** Located in `./model` directory

## Requirements

Install dependencies from the repository root:

```bash
pip install -r src/traffic_flow_gui/requirements.txt
```

## Required Artifacts

The app expects the following structure under `./model`:

```
./model/
├── output_network.xml                          (required)
├── models/
│   ├── best_ff_flowspeed_v3_globalcap.pth     (required - main model)
│   └── best_ff_flowspeed_v3_globalcap_gui.pth (optional - auto-generated if missing)
└── data/
    ├── artifacts_flowspeed_v3_globalcap.pkl   (required - feature engineering)
    ├── scalers_flowspeed_v3_globalcap.pkl     (required - normalization)
    └── model_configs_flowspeed_v3_globalcap.pkl (required - model configuration)
```

All artifacts except the optional `.pth` file must be present to run the GUI.

## Running the GUI

```bash
streamlit run src/traffic_flow_gui/app.py
```

The app will open in your browser at `http://localhost:8501` (or display the URL if automatic opening is disabled).

---

## User Interface Guide

### 1. **Top of Page - Model and Info**

The title bar displays the active model ("Traffic Flow & Speed Prediction") and basic instructions. Below that:
- **Model Source Info:** Shows which pre-trained model is loaded (e.g., "v3 global-capacity run")
- **Model Name:** Always "Feedforward NN" for this inference-only GUI
- **Speed Availability:** A status indicator showing whether speed predictions are available (requires `speed_scaler` in artifacts)

---

### 2. **Sidebar Controls** (Left Panel)

The sidebar contains all configuration and selection tools.

#### **Display Mode Section** 🎨

**Color arrows by:**
- **🚗 Flow (veh/hr)** – Color-code links by predicted vehicle flow
- **⚡ Speed (m/s)** – Color-code links by predicted traffic speed
  - *Speed mode is only available if the model artifacts include speed predictions*

**Map values** (three viewing options):
- **Absolute prediction** – Show raw predicted values
  - Flow: 0 (red) → medium (yellow) → high (green) veh/hr
  - Speed: 0 (red) → medium (yellow) → fast (green) m/s
- **Relative delta (pred - baseline)** – Show difference from baseline scenario
  - Red = lower than baseline → Blue = unchanged → Green = higher than baseline
  - Units: veh/hr for flow, m/s for speed
- **Relative percent (% vs baseline)** – Show percentage change from baseline
  - Red = lower than baseline → Blue = unchanged → Green = higher than baseline
  - Removes small noise near zero with a deadband filter

**Color Legend:**
The map displays a color legend below the arrows showing the current color-to-value mapping. The legend updates when you change display mode, value mode, or as you modify capacities.

#### **Base Scenario Section** 🧱

Choose a baseline capacity configuration for comparison. Available options:
- **1000 capacity all links** – Uniform 1000 veh/hr capacity
- **1800 capacity all links** – Uniform 1800 veh/hr capacity
- **3600 capacity all links** – Uniform 3600 veh/hr capacity (default, typical urban roads)
- **Custom base scenario** – Define your own per-link baseline capacities

When you select a predefined scenario, all link capacities reset to that uniform value. The baseline is used for calculating relative metrics. Switching scenarios clears cached predictions for real-time updates.

#### **Link Selection Section** 🔗

**Select Link** – Dropdown menu to choose which link to focus on:
- **Updates the time-series plot** (hourly flow or speed for the selected link)
- **Updates the flow vs. speed scatter** (if speed predictions are available)
- **Updates the local sensitivity analysis** (upstream/downstream interactions)
- **Highlights the selected link in blue on the map**

*Tip:* You can also select a link by **clicking directly on the map** (see Map Interaction below).

#### **Documentation Button** 📘

**Show documentation** – Opens this guide in a new browser tab for reference while using the GUI.

---

### 3. **Network Map** (Center/Top of Main Area)

Interactive pydeck-based map showing all links in the traffic network with color-coded directional arrows.

#### **Map Visual Elements**

- **Directional Chevrons** – Each link is drawn as a colored arrow pointing in the direction of traffic flow
- **Node Labels** – Numbered circles (1, 2, 3, ...) mark intersection nodes. The numbers match the "From" and "To" columns in the capacity table
- **Arrow Color Coding** – Color represents flow or speed based on your display mode selection
- **Selected Link Highlighting** – The currently selected link appears in **bright blue**
- **Blinking/Pulsing Effect** – When you select a link using the table's "Locate" checkbox, it briefly **pulses yellow** to draw your attention

#### **Hover Tooltips**

Hover your mouse over any arrow on the map to see a tooltip containing:
- **Link identifier** (e.g., "3-7")
- **Metric value** with units (e.g., "Flow Δ vs baseline: +120 veh/hr" or "Absolute prediction: 450 veh/hr")
- **Flow prediction** (veh/hr)
- **Speed prediction** (m/s and km/h)
- **Current capacity** (veh/hr)

#### **Map Interaction - Selecting a Link**

**To select a link by clicking:**

1. Click on any arrow or the link area on the map
2. A selection panel appears at the bottom-left with link details and two buttons:
   - **Confirm & Update** – Selects this link; all dependent views (charts, tables) update
   - **Cancel** – Closes the selection panel without making changes

*Note:* If map clicking doesn't work due to browser sandbox restrictions, use the **Select Link** dropdown in the sidebar instead.

**Navigating the map:**
- **Zoom:** Scroll wheel or pinch gesture
- **Pan:** Click and drag
- **Fit entire network:** Use the **Locate checkbox** in the capacity table (described below)

---

### 4. **Hour Slider**

Located just below the network map:

- **Range:** 5:00 to 22:00 (18 hours of traffic day, typical peak and off-peak hours)
- **Effect:** All predictions update in real time as you change the hour
- **Visual feedback:** The selected hour is highlighted in red on the time-series graph

---

### 5. **Link Capacities & Predictions Table**

Below the map, a comprehensive interactive table showing all links with their capacities and predicted metrics.

#### **Table Columns**

| Column | Description | Editable |
|--------|-------------|----------|
| **Locate** | Checkbox – check to select and pulse the link on the map, then fit the network to view | ✓ |
| **Modified** | Dot indicator (●) showing if this link's capacity differs from the base scenario | ✗ |
| **Link** | Simple identifier (e.g., "3-7" for link from node 3 to node 7) | ✗ |
| **From** | Source node number | ✗ |
| **To** | Destination node number | ✗ |
| **Base Capacity** | Baseline capacity used for relative comparisons (veh/hr) | ✓ |
| **Capacity** | Current/active capacity for predictions (veh/hr) | ✓ |
| **Base Flow** | Predicted flow at baseline capacity (veh/hr, scaled ×10) | ✗ |
| **Base Speed** | Predicted speed at baseline capacity (m/s) | ✗ |
| **Pred Flow** | Current predicted flow with active capacity (veh/hr, scaled ×10) | ✗ |
| **Pred Speed** | Current predicted speed with active capacity (m/s) | ✗ |

#### **Editing Capacities**

Click any **Base Capacity** or **Capacity** cell to edit:
- **Valid range:** 100 – 5000 veh/hr (in steps of 50)
- **Confirmation:** Press Enter or click away to apply; predictions update instantly
- **Effect on mode:**
  - In **Custom base scenario:** Editing "Base Capacity" resets the baseline for relative comparisons
  - In **predefined scenarios:** "Base Capacity" and "Capacity" are initially synced; editing either one independently is supported

#### **Table Sorting & Filtering**

**Sorting Controls:**
- **Sort mode selector:** Choose from "Modified first", "Link", "From-To", or sorting by any column (flow, speed, capacity)
- **Ascending/Descending toggle:** Reverse sort order (disabled when sorting by "Modified first")
- The current sort persists as you change the hour slider, so you can track specific links across time

**Finding Modified Links:**
- Click **Sort by: "Modified first"** to float all changed links to the top
- Look for the **●** indicator in the "Modified" column

#### **Reset All Button**

Click **↩️ Reset All** to:
- Reset all capacities back to the current base scenario values
- Clear all cached predictions to ensure fresh calculations
- Keep the selected link and display mode unchanged

---

### 6. **Selected Link vs Hour** (Time-Series)

Shows how the selected link's flow or speed changes across all 18 hours of the day.

**Components:**
- **Blue line with circular markers** – Predicted flow or speed at each hour under current capacity
- **Gray dashed line** – Baseline predictions (for comparison against original capacity)
- **Red dot** – The currently selected hour (from the hour slider)
- **X-axis:** Hour of day (5 to 22)
- **Y-axis:** Flow (veh/hr, scaled ×10) or Speed (m/s), depending on display mode

**Use:** Identify peak hours, off-peak patterns, and how capacity changes affect temporal distribution.

---

### 7. **Selected Link Flow vs Speed Scatter** (Congestion Relationship)

*Only shown if speed predictions are available.*

Displays the relationship between flow and speed for the selected link across all hours.

**Components:**
- **Blue dots** – One point per hour showing that hour's (flow, speed) pair
- **Red dot** – The currently selected hour (from the hour slider)
- **Gray line** – Chronological path connecting all hours (useful for seeing time-of-day progression)
- **Interactive:** Hover to see the hour; zoom and pan to explore specific regions

**Use:** Typically shows a negative relationship (higher flow → lower speed as congestion increases). Helps validate model predictions and understand link behavior.

---

### 8. **Local Upstream/Downstream Sensitivity Analysis** 🧭

At the bottom, a detailed three-panel sensitivity analysis for the selected link, revealing network effects.

#### **Layout & Components**

**Panel 1: Local Directional Subnetwork (Left)**
- Shows the selected link and its topologically connected neighbors
- **Center link** – Orange box (the link you selected)
- **Downstream links** – Green; connected INTO the center link's start node (traffic feeding into your link)
- **Upstream links** – Blue; connected OUT OF the center link's end node (traffic leaving your link)
- **Parallel links** – Dashed lines; alternate routes bypassing the center link
- Node labels match the map and table for easy cross-reference

**Panel 2: Flow Response Curve (Center)**
- **X-axis:** Center link capacity (swept from maximum down to minimum for visual clarity)
- **Y-axis:** Predicted flow (veh/hr, scaled ×10)
- **Multiple lines:** One for each link in the local subnetwork, color-coded by role (downstream, center, upstream, parallel)
- **Vertical black dashed line:** Marks the current capacity of the center link
- **Interpretation:** Steep slopes = high sensitivity; flat lines = low sensitivity to center link changes

**Panel 3: Speed Response Curve (Right)**
- Same layout as flow panel but showing speed (m/s) on the Y-axis
- *Only shown if speed predictions are available*
- Reveals how nearby links' speeds respond to center link capacity changes

#### **Sensitivity Hour Slider** (Independent)

Located above the three panels:
- **Range:** 5 to 22 (same as main hour slider)
- **Independent:** Allows you to analyze a different hour than the main map without changing the map view
- **Use:** Compare peak-hour sensitivity vs. off-peak sensitivity on the same link

#### **Interpretation Tips**

- **Steep upstream/downstream curves:** The center link is a bottleneck affecting traffic on adjacent links
- **Flat nearby curves:** The center link's capacity has little effect on its neighbors (decoupled from network)
- **Crossing curves:** Show complex interactions (e.g., relieving congestion on one downstream link while increasing congestion on another)
- **Upstream sensitivity:** High = your link creates backups upstream; Low = upstream is self-contained
- **Speed relationships:** Typically speeds drop as flow increases; dramatic speed drops indicate severe congestion

---

## Common Workflows

### **Explore Congestion Hotspots**

1. Use the hour slider to check different times of day
2. Look at the map for links showing red or very green (extreme values)
3. Select those links and examine their time-series and local sensitivity
4. Look for steep curves in the sensitivity analysis indicating bottlenecks

### **Analyze Network Effects**

1. Select a link in the middle of the network
2. Look at the local sensitivity analysis to see upstream and downstream links
3. Increase the selected link's capacity and watch how upstream/downstream links respond
4. Use the sensitivity hour slider to compare effects at different times of day

### **Compare Scenarios**

1. Choose a base scenario (e.g., "1800 capacity all links")
2. Switch to **Relative percent** value mode for easy comparison
3. Identify which links deviate most from the baseline
4. Select high-deviation links to understand why

### **What-If Analysis**

1. Start with a base scenario (e.g., "3600 capacity all links")
2. Modify specific link capacities in the table (e.g., increase a problematic intersection)
3. Watch the map colors change and flow values update
4. Check the local sensitivity analysis to see network-wide impact
5. Use **Locate** on modified links to highlight them on the map

### **Validate Model Predictions**

1. Set all links to a uniform capacity (predefined scenario)
2. Observe the predicted flow distribution at different hours
3. Compare against known empirical data or traffic counts
4. Look for unrealistic patterns (e.g., capacity underutilization or flow conservation violations)

---

## Important Notes

### **Flow Scaling (Critical Detail)**

Flow values displayed in the GUI are **scaled by 10× for visualization**. This accounts for the MATSim model using a 10% sample population:
- **Displayed value:** 500 veh/hr = **Actual prediction:** 50 veh/hr
- **Scaling applied to:** All flow displays (map, table, time-series graph, local sensitivity)
- **Why:** Helps users understand full-population equivalents (what the actual traffic would be if all vehicles were included)

### **Baseline vs. Current**

- **Baseline** (gray dashed line in time-series, "Base" columns in table) – Fixed reference from the selected base scenario
- **Current** (blue line in time-series, "Pred" columns in table) – Updates in real time as you edit capacities

When you switch base scenarios, the baseline predictions reset but your current capacity edits may remain (depending on scenario type).

### **Relative Views & Deadband Filter**

- **Relative delta:** Shows difference in same units as absolute (veh/hr or m/s)
- **Relative percent:** Shows percentage change; easier to compare across different-sized links
- **Deadband filter:** Small changes near zero (e.g., ±0.5%) are filtered out to reduce visual noise in relative views

### **Speed Availability**

Speed predictions are only available if the model artifacts include a trained speed model. Check the status indicator at the top of the page:
- ✅ If available → Speed mode and scatter plot are enabled
- ⚠️ If unavailable → Only flow predictions shown; speed-related controls disabled

This depends on whether the `speed_scaler` and output_dim=2 model configuration are present in the artifacts.

### **Caching & Performance**

- The app caches predictions to avoid redundant model inference
- Caches are automatically cleared when you:
  - Change the base scenario
  - Edit link capacities
  - Access new features requiring recomputation
- For large networks, initial load or major changes may take a few seconds

### **Inference-Only Architecture**

- No model training happens in this GUI
- All predictions use the pre-trained feedforward model from `./model`
- Model training and retraining instructions are in `README_MODEL_TRAINING.md`

### **Network Effects & Global Capacity Context**

The feedforward model is trained with "global capacity context," meaning predictions account for:
- The selected link's local capacity
- Upstream/downstream neighbor capacities
- Full-network capacity distribution
- Hour of day and link static features (geometry, connectivity)

This is why the local sensitivity analysis shows non-trivial upstream/downstream responses—the model learned realistic network propagation.

---

## Keyboard & Browser Tips

- **Full-screen map:** Use browser DevTools (F12) to inspect the map or adjust zoom
- **Map link selection fallback:** If map clicking doesn't work due to browser sandbox, use the **Select Link** dropdown in the sidebar
- **Table export:** Right-click the table and use your browser's "Print" or "Save as PDF" to export data
- **Undo edits:** Use the **↩️ Reset All** button to revert all capacity changes at once
- **Link sharing:** Copy the browser URL (including query parameters) to share a specific view or link selection

---

## Troubleshooting

**Speed predictions not showing:**
- Check that `scalers_flowspeed_v3_globalcap.pkl` and `model_configs_flowspeed_v3_globalcap.pkl` are present in `./model/data/`
- Verify the model config includes `output_dim: 2` for dual-task (flow+speed) predictions

**Map clicking not selecting links:**
- This is a browser sandbox restriction; use the **Select Link** dropdown in the sidebar instead

**Predictions seem unrealistic:**
- Check that the selected hour is within 5–22 (outside this range, predictions may be extrapolated)
- Verify capacity values are within the 100–5000 veh/hr range (model was trained on this domain)

**Large network is slow:**
- The model runs CPU inference by default; check available memory if network is very large
- Predictions are cached; they speed up after the first request for each unique capacity configuration

---

## Next Steps

- Review `README_MODEL_TRAINING.md` for instructions on training new models
- Explore `README.md` for project overview and structure
- Check `notebooks/` for detailed analysis examples and model evaluation

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Map clicking doesn't work | Try using the **Link Selection** dropdown in the sidebar instead |
| Speed predictions not shown | Check that speed model artifacts are present in `./model/data/` |
| Predictions seem wrong | Verify base scenario is set correctly; check that model artifacts are up-to-date |
| App is slow | The network may be large; wait for initial load, or focus on fewer links |
| Documentation doesn't open | Check browser pop-up blocker settings; allow pop-ups for this site |

---

## See Also

- [README_MODEL_TRAINING.md](README_MODEL_TRAINING.md) – Model training and retraining instructions
- [README.md](README.md) – Overall project documentation

# Traffic Flow & Speed prediction

The published GUI is intentionally restricted to one fixed, pre-trained model:

- Feedforward Neural Network (FF) only
- No model chooser dialog
- No model switching controls
- Inference-only usage

See `README_GUI.md` for full GUI setup and run instructions.

## Run the GUI

```bash
pip install -r src/traffic_flow_gui/requirements.txt
streamlit run src/traffic_flow_gui/app.py
```

## Data and Artifacts

The GUI uses `./model/output_network.xml` and pre-trained FF artifacts under `./model`.

## Training History

Training code was removed from this publish branch, but the historical training process is documented in `README_MODEL_TRAINING.md`.

## Planned further work
The present public GUI is limited to the feedforward neural-network model described above. Planned further work is to extend the traffic-prediction workflow with Graph Neural Network (GNN) models that better exploit the road-network structure.
This development is connected to the GNN modelling work in Grunde Wesenberg's PhD project at the University of Bergen, carried out as part of PRELONG. Once the GNN-based modelling work is published and sufficiently validated, future versions of the GUI may include graph-based model backends, improved use of network topology, and richer scenario handling. 

## Funding
This work was developed as part of the PRELONG project, Machine learning for computational efficient predictions of long-term congestion patterns in large-scale transport systems, funded by the Research Council of Norway (NFR), project ID 322480.
PRELONG investigates how machine learning can be combined with agent-based traffic simulation to support fast and user-friendly prediction of long-term congestion patterns in large-scale transport systems.
## Acknowledgements
The GUI and associated traffic-assignment prediction workflow build on research and development carried out in PRELONG. The original idea for an AI-based traffic-planning tool was formulated in the public PRELONG webinar/presentation Building an AI-based tool for traffic planning on 7 September 2021, where the central question was whether city-wide traffic flow and congestion patterns could be predicted quickly using machine-learning models rather than repeatedly running full-scale simulation models.
We acknowledge the main contributors to the GUI, modelling workflow, and underlying simulations:
- Grunde Wesenberg
- Anna Piterskaya
- Christian Weber
- Stefan Flügel
The work also builds on the synthetic population and synthetic travel-demand modelling documented in TØI Report 2065/2024, which describes the generation of activity plans for a synthetic population in the Greater Oslo Area using machine-learning models trained on Ruter-MIS travel survey data and applied to population and commuting data.
## References
Flügel, S. (2021, September 7). _Building an AI-based tool for traffic planning: Project outline_ [Public webinar presentation]. Institute of Transport Economics (TØI). https://www.youtube.com/watch?v=6TltoKAnVcE

Flügel, S., Weber, C., Klommestein, S. S., Korsmo, J., & Kielland, A. (2024). _Towards activity-based demand modelling for the Greater Oslo Area: Using machine learning to predict travel mode choice and activity plans_ (TØI Report 2065/2024). Institute of Transport Economics.
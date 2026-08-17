## Knowledge Graph Analysis of the ICIJ Offshore Leaks
Project exploring knowledge graph (KG) techniques on real financial crime data, using the ICIJ Offshore Leaks dataset (Panama Papers, Paradise Papers, Bahamas Leaks).

The goal is to learn how to use KG to reveal connections that are invisible in a data table, and to understand why graph databases are a tool of choice for financial intelligence work.

## Key Finding
Portcullis TrustNet BVI is the most connected and most influential node, with 37,338 shell companies registered to this address in the British Virgin Islands.

## Dataset
I will use the ICIJ Offshore Leaks dataset comprising of the Panama Papers, Paradise Papers, Bahamas Leaks and Offshore Leaks investigations obtained from Kaggle:

kaggle datasets download -d zusmani/paradisepanamapapers -p /tmp/paradise --unzip

The data covers more than 810,000 offshore entities and the people and intermediaries connected to them.
1,040,535 nodes and 1,535,552 relationships across Panama Papers, Bahamas Leaks and Offshore Leaks investigations.

## Notebooks
01_EDA - explore data table to understand data structure, distributions, null rates
02_graph_analysis - building graph using NetworkX, graph analysis, degree centrality, PageRank and PyVis visualisation

## Visualisation
Open `notebooks/KG_Paradise_Visual.html` in browser to explore the interactive graph.

## Stack
pandas, NetworkX, PyVis, matplotlib, seaborn

## Data Source
International Consortium of Investigative Journalists (ICIJ) -- https://offshoreleaks.icij.org
via Kaggle:
Licensed under the Open Database License.
Always cite ICIJ when using this data.

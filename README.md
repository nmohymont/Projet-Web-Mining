# Comparative analysis of descriptive lexicons : the positioning of lgobal universities according to two ranking organizations.

## Goal of the project
This project analyzes how international universities are described by the two major global ranking organization known as Quacquarelli Symonds (QS) and Times Higher Education (THE). It identifies the main themes and relationships present in their descriptive texts and assess how lexical differencees shape the positioning and public image of universities. The study shows that these descriptions are not neutral: they function as communication tools that contribute to academic branding strategies. 

## Guidelines to use this code 

### Mandatory preliminary phase

Download the HTML files here:
[Download HTML dataset from Dropbox](https://www.dropbox.com/scl/fi/kgtgrrf4rx30w5swwxg8l/HTML_web_scraping.zip?rlkey=wjjxeo3p9zaam34sayrq0a6gy&st=nouuemzh&dl=0 ) to download all html files needed for the web scraping. Upload those files to this directory DATA/CLEAN/HTML. This step is important because the source code of the Times Higher Education websites collected thanks to a simple requests.get() command is not sufficient as the website loads dynamic content while scrolling though the table that displays the university ranking. 

### Web Scraping
After the mandatory step, the code begins with the web scraping phase in  ```web_scrapping.ipynb```. The ranking websites are analyzed to extract the descriptive texts associated with each university. The notebook is divided into four sections, corresponding to the four scraped websites : two sections are dedicated to the 2025 rankings of QS and THE, and two others are archived THE websites from 2012 and 2021, used for temporal comparison. 

The scraping process extracts the **university name, ranking position, word region and descriptive text**. These data are savec as CSV and PARQUET files, which are alreday provided in the repository so that you do not need to run the scraping code for several hours. The collected datasets were produced and used strictly for **academix purposes**, whitin the framework of a university course in web mining. 

### Additional data cleaning step

Despite an initial preprocessing stage that was not fully adapted to our analysis, a ```data_cleaning``` folder includes two additional scripts.

The first script,```clean_qs_descriptions_pyenchant.py```, uses the **pyenchant** libarry to clean the QS descriptions collected during scraping, since spacing errors and incorrectly segmented words were observed. The library automatically corrects words using an Engligh dictionary. It is important to note that pyenchant is **not supported in Python 3.13**, we used Python 3.11 to correct 2 843 terms across 660 descriptions out of the 1140 collected. 

The second script, ```names_matching_qs_the.py```, uses the **difflib** library to compute a similarity score between university names taht are not exactly identical. For example, it allows "Université **l**ibre de Bruxelles" et " Université **L**ibre de Bruxelles" to be considered the same institution, with a similarity score of 0.96.

### Text Mining

**Note : it is important to uncomment the ```nltk.download``` commands at the beginning of each script using the NLTK libary during the execution. This only needs to be run once during the first execution.**

#### Dimensionnality testing
The script ```text_mining_dimensionality_test.py``` gathers all prelimnary text mining tests performed on the collected descriptions to justify the selection of :
* Normalisation methods : Stemming vs Lemmatization
* Frequency filtering tresholds 
* Vectorization methods : TF-IDF vs BERT
* Similarity metrics : Cosine vs Jaccard

At the begining of the script, a mode selector variable named ```CURRENT_MODE``` allows to swtich easily between different PARQUET files. At the end of this script, a JSON file is created containgin an ordered selection of the best TF-IDF lemmatized tokens, the region assiocated with each university, and their ranking positions. These JSON file store all preprocessed tokens required for further analysis : text mining applications and link analysis.

Important: These script must be executed at least once to generate the necessary JSON file. The subsequent descriptive and semantic analyses strictly rely on these JSON output to function."
#### Text Mining Applications

habiabbi 
A ```text_mining_applications``` folder includes five scripts dedicated to different types of analysis such as :
* Comparative analysis of descriptions provided  by QS and THE from their 2025 rankings
* Descriptive analysis : 
    * Word clouds 
    * Co-occurrence graph
    * Temporal analysis 
* Semantic analysis :
    * Bar chart of dominant feelings
    * Radar plot based on all recorded feelings
    * Regional trends  
* Clustering analysis 

### Link Analysis
The script ```link_analysis.py``` generates graphs from preprocessed tokens (lemmatized and ranked by importance using TF-IDF). Two approaches are used :
* Undirected graph : This graph connects tokens using the Jaccard similarity index. This method highlights commonly associated tokens while reducing the influence of highly frequent tokens. It is used for centrality analysis.
* Artificially directed graph : A second graph is generated based on the 5-Nearest-Neighbors of each token. To avoid bidirectional links, a specific rule is applied : the edge is directed from the less frequent node to the more frequent one. This structure allows the application of prestige algorithms such as HITS and PageRank.

Both graphs are exported in ```.gexf``` format for visualization and analysis in Gephi. The scores computed in the script and those obtained in Gephi are identical, confirming the validity of the measures and conclusions produced in the script. 
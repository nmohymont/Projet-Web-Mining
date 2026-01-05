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

Despite an initial preprocessing stage that was not fully fully adapted to our analysis, a ```data_cleaning``` folder includes two additional scripts.

The first script,```clean_qs_descriptions_pyenchant.py```, uses the **pyenchant** libarry to clean the QS descriptions collected during scraping, since spacing erriors and incorrectly segmented words were observed. The library automatically corrects words words using an Engligh dictionary. It is important to note that pyenchant is **not supported in Python 3.13**, we used Python 3.11 to correct 2 843 terms across 660 descriptions out of the 1140 collected. 

The second script, ```names_matching_qs_the.py```, uses the **difflib** library to compute a similarity score between university names taht are not exactly identical. For example, it allows "Université **l**ibre de Bruxelles" et " Université **L**ibre de Bruxelles" to be considered the same institution, with a similarity score of 0.96.

### Text Mining

**Note : it is important to uncomment the nltk.download steps at the begining of each scripts that uses the nltk libary during the first use of the script**

#### Dimensionnality test
The script ```text_mining_dimensionality_test.py``` gathers all the prelimnary text mining test used on the collected descriptions to justify the use of :
* normalisation methods : stemmer vs lemmatizer
* frequency filtering tresholds 
* vectorisation methods : TF-IDF vs BERT
* similarity metrics : cosine VS jaccard

At the begining of the script a selector mode is named under the variable ```CURRENT_MODE``` to swtich easily from one PARQUET file to another. At the end of this script, a JSON file is created with a ordered selection of the best TF-IDF lemmatized token, the region assiocated to each university and its ranking positions. These JSON file stored all the preprocessed tokens that are needed for the further analysis : text mining applications and the link analysis.

#### Text Mining Applications

A ```text_mining_applications``` folder includes five scripts to different type of analysis such as :
* comparison of the description given by QS and THE from their 2025's ranking
* descriptive analysis 
    * word clouds 
    * co-occurrence graph
    * temporal analysis 
* semantic analysis
    * bar chart with dominant feelings
    * radar plot based on all feelings recorded
    * regional trends  
* clustering analysis 

### Link Analysis
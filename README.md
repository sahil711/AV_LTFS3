# AV LTFS3 FINHACK
### <i>link: https://datahack.analyticsvidhya.com/contest/ltfs-data-science-finhack-3</i>

<b><i>This repo contains the 2nd place solution. </b></i>

Steps to replicate the final solution:
- <b>Advice</b>: 
    - place the data in a folder inside the same directory (we will assume the name to be 'data/')
    - when running notebooks, change the `DATA-DIR` variable to point to the path where the raw data is kept
1. `pip install -r requirements.txt`
2. `bash run.sh -p 'data/'` ('data/' is the folder where the raw data is present)
3. Run final_lgbm_seed2.ipynb
4. Run final_lgbm_seed122.ipynb 
5. Run final_lgbm_seed777.ipynb
6. Run final_blend.ipynb

<b> Output </b>: final_submission.csv (scores 84.95 on Public LB and 83.5 on Private LB) 


library(dplyr)    
library(tidyr)

# Data Load ----
url <- "https://github.com/dutangc/CASdatasets/blob/master/data/freMTPL2freq.rda?raw=true"
load(url(url))
url <- "https://github.com/dutangc/CASdatasets/blob/master/data/freMTPL2sev.rda?raw=true"
load(url(url))


#########################################################################################################################
# Will have 4 sets of data: Original, Transformed features, Truncated Original, Truncated Transformed Features #
#########################################################################################################################

# For each set we will have 3 data objects:
  # info - dataset to train the number of claims model
  # comb - dataset to train the amount of a claim model
  # comb_agg - dataset to train the total amount for a policy
   
  # Then to split the datasets we split the info and comb dataset
    # Then in order to get the conformal intervals and check the coverage for the total claims we use the same split as used for the info set

  # Using stratified sampling to generate the 3 sets

#### Original ####
info = freMTPL2freq ; claims = freMTPL2sev
rm(freMTPL2freq);rm(freMTPL2sev)
# removing outliers
# for the info dataset truncated the number of claims to 4
info$ClaimNb_cap = pmin(info$ClaimNb, 4)
info = select(info, -"ClaimNb")
# for the claims dataset have removed the top 5% of values
claims = claims[claims$ClaimAmount < quantile(claims$ClaimAmount, 0.95),]
# Need to correct the number of claims for each policy now for the combined and aggragated dataset.
claims = claims %>%
group_by(IDpol) %>%
mutate(ClaimNb = n()) %>%
ungroup()

comb = merge(x = info, y = claims, by = "IDpol", all.x = TRUE)
comb$ClaimAmount[is.na(comb$ClaimAmount)] = 0
comb = select(comb, -"ClaimNb_cap")
comb$ClaimNb = ifelse(is.na(comb$ClaimNb), 0, comb$ClaimNb)
comb$ClaimNb_cap = pmin(comb$ClaimNb, 4)

comb_agg = comb %>%
  group_by(IDpol) %>%
  summarise(ClaimAmount = sum(ClaimAmount))

comb = comb[!comb$ClaimAmount == 0,]

# stratified sampling
  # info dataset
zeroind = which(info$ClaimNb == 0)
nonzeroind = which(info$ClaimNb > 0)


set.seed(100)
traincalzeroind = sample(zeroind, floor(0.8*length(zeroind)))
testzero = info[-traincalzeroind,] 
calzeroind = sample(traincalzeroind, floor(0.0625*length(zeroind)))
calzero = info[calzeroind,] 
trainzero = info[setdiff(traincalzeroind,calzeroind),] 

traincalnzind = sample(nonzeroind, floor(0.8*length(nonzeroind)))
testnz = info[-traincalnzind,] 
calnzind = sample(traincalnzind, floor(0.0625*length(nonzeroind)))
calnz = info[calnzind,] 
trainnz = info[setdiff(traincalnzind,calnzind),] 

info_train = rbind(trainzero, trainnz)
info_cal = rbind(calzero, calnz)
info_test = rbind(testzero, testnz)




##################


#### Transformed features ####
info_tr = info; claims_tr = claims

# Feature pre-processing for GLM regression
info_tr$AreaGLM <- as.integer(info_tr$Area)
info_tr$VehPowerGLM <- as.factor(pmin(info_tr$VehPower,9))
VehAgeGLM <- cbind(c(0:110), c(1, rep(2,10), rep (3,100)))
info_tr$VehAgeGLM <- as.factor(VehAgeGLM[info_tr$VehAge+1,2])
info_tr[,"VehAgeGLM"] <- relevel(info_tr[,"VehAgeGLM"], ref="2")
DrivAgeGLM <- cbind(
  c(18:100),  # Age vector from 18 to 100
  c(
    rep(1, 4),    # Ages 18-21 (4 values)
    rep(2, 5),    # Ages 22-26 (5 values)
    rep(3, 5),    # Ages 27-31 (5 values)
    rep(4, 10),   # Ages 32-41 (10 values)
    rep(5, 10),   # Ages 42-51 (10 values)
    rep(6, 20),   # Ages 52-71 (20 values)
    rep(7, 29)    # Ages 72-100 (29 values)
  )
)
info_tr$DrivAgeGLM <- as.factor(DrivAgeGLM[info$DrivAge-17,2])
info_tr[,"DrivAgeGLM"] <- relevel(info_tr[,"DrivAgeGLM"], ref = "5")
info_tr$BonusMalusGLM <- as.integer(pmin(info_tr$BonusMalus, 150))
info_tr$DensityGLM <- as.numeric(log(info_tr$Density))
info_tr[,"Region"] <- relevel(info_tr[,"Region"], ref="Centre")
##################################





## Data Edit ----
# Merge datasets based on matching IDpol
# When merging datasets, duplicate rows are created for those who have made more then 1 claim.
freMTPL2_Amount_per_claim <- merge(freMTPL2freq, freMTPL2sev, by = "IDpol", all.x = TRUE)

# This data set is used to model amount of claims
freMTPL2_Amount_per_claim <- freMTPL2_Amount_per_claim[, names(freMTPL2_Amount_per_claim) != "ClaimNb"]

# Group freMTPL2sev by IDpol and sum ClaimAmount
freMTPL2sev_Claim_number <- freMTPL2sev %>%
  group_by(IDpol) %>%
  summarise(ClaimAmount = sum(ClaimAmount))

#This Dataset is used to model Amount per claim
freMTPL2_Claim_number <- merge(freMTPL2freq, freMTPL2sev_Claim_number, by = "IDpol", all.x = TRUE)

# Create new variable with any claimNb over 4 grouped at 4
freMTPL2_Claim_number$ClaimNb_cap <- pmin(freMTPL2_Agd$ClaimNb, 4)
freMTPL2freq$ClaimNb_cap <- pmin(freMTPL2freq$ClaimNb, 4)
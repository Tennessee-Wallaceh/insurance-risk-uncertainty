library(dplyr)    
library(tidyr)

# Data Load ----
url <- "https://github.com/dutangc/CASdatasets/blob/master/data/freMTPL2freq.rda?raw=true"
load(url(url))
url <- "https://github.com/dutangc/CASdatasets/blob/master/data/freMTPL2sev.rda?raw=true"
load(url(url))


#########################################################################################################################
# Will have 2 sets of data: Original, Transformed features #
#########################################################################################################################

# For each set we will have 3 data objects:
  # info - dataset to train the number of claims model
  # comb - dataset to train the amount of a claim model
  # comb_agg - dataset to test model for modelling the total amount for a policy
   
  # Then to split the datasets we split the info and comb dataset
    # Then in order to get the conformal intervals and check the coverage for the total claims we use the same split as used for the info set

  # Using stratified sampling to generate the 3 sets

#### Original ####
freMTPL2freq$Exposure = pmin(freMTPL2freq$Exposure, 1)
info = freMTPL2freq ; claims = freMTPL2sev
# rm(freMTPL2freq);rm(freMTPL2sev)
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

  
# comb dataset, not stratified sampling as only using data with non zero claims
set.seed(100)
traincalind = sample(1:nrow(comb), floor(0.8*(nrow(comb))))
comb_test = comb[-traincalind,] 
calind = sample(traincalind, floor(0.0625*nrow(comb)))
comb_cal = comb[calind,] 
comb_train = comb[setdiff(traincalind,calind),] 
 


##########################################################################################################################################################################


#### Transformed features ####
info_tr = freMTPL2freq; claims_tr = freMTPL2sev

# Feature pre-processing for GLM regression
info_tr$Area <- as.integer(info_tr$Area)
# truncating vehicle power to 9
info_tr$VehPower <- as.factor(pmin(info_tr$VehPower,9))
# Creating 3 groups for the Vehicl age
VehAge <- cbind(c(0:110), c(1, rep(2,10), rep (3,100)))
info_tr$VehAge <- as.factor(VehAge[info_tr$VehAge+1,2])
info_tr[,"VehAge"] <- relevel(info_tr[,"VehAge"], ref="2")
# Creating 7 groups for the driver age
DrivAge <- cbind(
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
info_tr$DrivAge <- as.factor(DrivAge[info$DrivAge-17,2])
info_tr[,"DrivAge"] <- relevel(info_tr[,"DrivAge"], ref = "5")
# trunacating BonusMalus to 150
info_tr$BonusMalus <- as.integer(pmin(info_tr$BonusMalus, 150))
# log transform for density
info_tr$Density <- as.numeric(log(info_tr$Density))
info_tr[,"Region"] <- relevel(info_tr[,"Region"], ref="Centre")
##################

info_tr$ClaimNb_cap = pmin(info_tr$ClaimNb, 4)
info_tr = select(info_tr, -"ClaimNb")
# for the claims dataset have removed the top 5% of values
claims_tr = claims_tr[claims_tr$ClaimAmount < quantile(claims_tr$ClaimAmount, 0.95),]
# Need to correct the number of claims for each policy now for the combined and aggragated dataset.
claims_tr = claims_tr %>%
  group_by(IDpol) %>%
  mutate(ClaimNb = n()) %>%
  ungroup()

comb_tr = merge(x = info_tr, y = claims_tr, by = "IDpol", all.x = TRUE)
comb_tr$ClaimAmount[is.na(comb_tr$ClaimAmount)] = 0
comb_tr = select(comb_tr, -"ClaimNb_cap")
comb_tr$ClaimNb = ifelse(is.na(comb_tr$ClaimNb), 0, comb_tr$ClaimNb)
comb_tr$ClaimNb_cap = pmin(comb_tr$ClaimNb, 4)

comb_agg_tr = comb_tr %>%
  group_by(IDpol) %>%
  summarise(ClaimAmount = sum(ClaimAmount))

comb_tr = comb_tr[!comb_tr$ClaimAmount == 0,]

# stratified sampling
# info dataset
info_tr$VehPower = as.numeric(info_tr$VehPower)
info_tr$VehAge = as.numeric(info_tr$VehAge)
info_tr$DrivAge = as.numeric(info_tr$DrivAge)

testzero = info_tr[-traincalzeroind,] 
calzero = info_tr[calzeroind,] 
trainzero = info_tr[setdiff(traincalzeroind,calzeroind),] 

testnz = info[-traincalnzind,] 
calnz = info[calnzind,] 
trainnz = info[setdiff(traincalnzind,calnzind),] 

info_tr_train = rbind(trainzero, trainnz)
info_tr_cal = rbind(calzero, calnz)
info_tr_test = rbind(testzero, testnz)



# comb dataset, not stratified sampling as only using data with non zero claims
comb_tr_test = comb_tr[-traincalind,] 
comb_tr_cal = comb_tr[calind,] 
comb_tr_train = comb_tr[setdiff(traincalind,calind),] 




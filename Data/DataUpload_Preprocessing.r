# Data Load ----
url <- "https://github.com/dutangc/CASdatasets/blob/master/data/freMTPL2freq.rda?raw=true"
load(url(url))
url <- "https://github.com/dutangc/CASdatasets/blob/master/data/freMTPL2sev.rda?raw=true"
load(url(url))

# Feature pre-processing for GLM regression
freMTPL2freq$AreaGLM <- as.integer(freMTPL2freq$Area)
freMTPL2freq$VehPowerGLM <- as.factor(pmin(freMTPL2freq$VehPower,9))
VehAgeGLM <- cbind(c(0:110), c(1, rep(2,10), rep (3,100)))
freMTPL2freq$VehAgeGLM <- as.factor(VehAgeGLM[freMTPL2freq$VehAge+1,2])
freMTPL2freq[,"VehAgeGLM"] <- relevel(freMTPL2freq[,"VehAgeGLM"], ref="2")
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
freMTPL2freq$DrivAgeGLM <- as.factor(DrivAgeGLM[freMTPL2freq$DrivAge-17,2])
freMTPL2freq[,"DrivAgeGLM"] <- relevel(freMTPL2freq[,"DrivAgeGLM"], ref = "5")
freMTPL2freq$BonusMalusGLM <- as.integer(pmin(freMTPL2freq$BonusMalus, 150))
freMTPL2freq$DensityGLM <- as.numeric(log(freMTPL2freq$Density))
freMTPL2freq[,"Region"] <- relevel(freMTPL2freq[,"Region"], ref="Centre")

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

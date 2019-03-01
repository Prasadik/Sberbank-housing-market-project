#Remove all objects from R workspac
rm(list=ls())

#Set current working directory where the data_set files are stored
#setwd("C:/Users/Desktop/Project1_Prasad_IK/Data_Set")

#Load packages required to perform certain functions
library(Matrix)
library(caret)
library(data.table)
library(xgboost)

#Load the data into R using fread
data_train=fread("train.csv", sep=",", na.strings = "NA",verbose = FALSE)
data_test=fread("test.csv", sep=",", na.strings = "NA",verbose = FALSE)


#Transform the target variable, so that we can use RMSE in XGBoost
data_train$price_doc=log1p(as.integer(data_train$price_doc))
data_test$price_doc="-1"

#Combine the tables data_train and data_test
data=rbind(data_train,data_test)


#Convert the characters types to factors
data$product_type=as.factor(data$product_type)
data$sub_area=as.factor(data$sub_area)
data$ecology=as.factor(data$ecology)
data$thermal_power_plant_raion=ifelse(data$thermal_power_plant_raion=="no",0,1)
data$incineration_raion=ifelse(data$incineration_raion=="no",0,1)
data$oil_chemistry_raion=ifelse(data$oil_chemistry_raion=="no",0,1)
data$radiation_raion=ifelse(data$radiation_raion=="no",0,1)
data$railroad_terminal_raion=ifelse(data$railroad_terminal_raion=="no",0,1)
data$big_market_raion=ifelse(data$big_market_raion=="no",0,1)
data$nuclear_reactor_raion=ifelse(data$nuclear_reactor_raion=="no",0,1)
data$detention_facility_raion=ifelse(data$detention_facility_raion=="no",0,1)
data$culture_objects_top_25=ifelse(data$culture_objects_top_25=="no",0,1)
data$water_1line=ifelse(data$water_1line=="no",0,1)
data$big_road1_1line=ifelse(data$big_road1_1line=="no",0,1)
data$railroad_1line=ifelse(data$railroad_1line=="no",0,1)


#Convert the variable timestamp to Date type and hence produce additonal features
data$timestamp=as.Date(data$timestamp)
data$date_month=month(data$timestamp)
data$date_year=year(data$timestamp)
data$date_week=week(data$timestamp)

data$date_yearmonth=data$date_year*100+data$date_month
data$date_yearweek=data$date_year*100+data$date_week


#na_rowCount gives the sum of NAs row-wise
data$na_rowcount=rowSums(is.na(data))


#Some additional features useful for training the model
#data$kitchen_life_ratio=data$kitch_sq/data$life_sq
#data$floor_ratio=data$floor/data$max_floor
data$life_sq_ratio=data$life_sq/data$full_sq
data$kitchen_sq_ratio=data$kitch_sq/data$full_sq
data$sq_per_floor_ratio=data$full_sq/data$floor
data$building_lifetime=data$date_year - data$build_year



#Load macro data and merge the data with data_macro
data_macro=fread("macro.csv", sep=",", na.strings = "NA",verbose = FALSE)
data_macro$timestamp= as.Date(data_macro$timestamp)
data_macro=sapply(data_macro,as.numeric) 
data=merge(data, data_macro, by="timestamp", all.x=TRUE)
gc() #garbage collection


#To produce sparse matrix
var_names=setdiff(colnames(data), c("id", "price_doc", "timestamp"))
train_sparse=Matrix(as.matrix(sapply(data[price_doc > -1, var_names, with=FALSE],as.numeric)), sparse=TRUE)
test_sparse=Matrix(as.matrix(sapply(data[price_doc == -1, var_names, with=FALSE],as.numeric)), sparse=TRUE)
y_train=data[price_doc > -1,price_doc]
test_ids=data[price_doc == -1,id]
dtrain=xgb.DMatrix(data=train_sparse, label=y_train)
dtest=xgb.DMatrix(data=test_sparse)
gc() #garbage collection

#Parameters required for xgboost model
parameters=list(objective="reg:linear",
                eval_metric="rmse",
                eta=0.05,
                gamma=1,
                max_depth=4,
                min_child_weight=1,
                subsample=0.7,
                colsample_bytree =0.7
)

rounds=400

#Train the model using XGB
xgb_model=xgb.train(data=dtrain,
                       params=parameters,
                       watchlist=list(train = dtrain),
                       nrounds=rounds,
                       verbose=1,
                       print_every_n=5
)
gc() #garbage collection


#Variable importance: plotting top 24 important variables
names=dimnames(train_sparse)[[2]]
important_variables=xgb.importance(names,model=xgb_model)
xgb.plot.importance(important_variables[1:24,])

# Predict the model and thus output the submission.csv file
pred=predict(xgb_model,dtest)
pred=expm1(pred)
write.table(data.table(id=test_ids, price_doc=pred), "submission.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)

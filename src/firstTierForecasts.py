'''
File to generate source production forecasts for different sources across regions.
'''

import csv
from datetime import datetime as dt
from datetime import timezone as tz

import numpy as np
import pandas as pd
import pytz as pytz
from keras.layers import Dense, Flatten
from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import common
import sys
import json5 as json


############################# MACRO START #######################################
BUFFER = -1
DAY_INTERVAL = 1
MONTH_INTERVAL = 1
DEPENDENT_VARIABLE_COL = 0

TRAINING_WINDOW_HOURS = None
PREDICTION_WINDOW_HOURS = None
MODEL_SLIDING_WINDOW_LEN = None
BUFFER_HOURS = None

############################# MACRO END #########################################

def runFirstTier(configFileName):
    global TRAINING_WINDOW_HOURS
    global PREDICTION_WINDOW_HOURS
    global MODEL_SLIDING_WINDOW_LEN
    global BUFFER_HOURS

    firstTierConfig = {}

    with open(configFileName, "r") as configFile:
        firstTierConfig = json.load(configFile)
        # print(configurationData)

    NUMBER_OF_EXPERIMENTS = firstTierConfig["NUMBER_OF_EXPERIMENTS_PER_REGION"]
    TRAINING_WINDOW_HOURS = firstTierConfig["TRAINING_WINDOW_HOURS"]
    PREDICTION_WINDOW_HOURS = firstTierConfig["PREDICTION_WINDOW_HOURS"]
    MODEL_SLIDING_WINDOW_LEN = firstTierConfig["MODEL_SLIDING_WINDOW_LEN"]
    BUFFER_HOURS = PREDICTION_WINDOW_HOURS - 24

    regionList = firstTierConfig["REGION"]
    for region in regionList:
        print("CarbonCast: ANN model for region:", region)
        regionConfig = firstTierConfig[region]
        sourceList = regionConfig["SOURCES"]
        sourceColList = regionConfig["SOURCE_COL"]
        trainTestPeriodConfig = firstTierConfig["TRAIN_TEST_PERIOD"]
        weatherForecastInFileName = regionConfig["WEATHER_FORECAST_IN_FILE_NAME"]
        SAVED_MODEL_LOCATION = firstTierConfig["SAVED_MODEL_LOCATION"]+region+"/"
        aggregatedForecastFileName = regionConfig["AGGREGATED_FORECAST_OUT_FILE_NAME"]


        sourceIdx = 0
        outFileNamePrefix = regionConfig["OUT_FILE_NAME_PREFIX"]
        for source in sourceList:
            inFileName = regionConfig["IN_FILE_NAME_PREFIX"] + source.lower() + firstTierConfig["IN_FILE_NAME_SUFFIX"]
            sourceCol = sourceColList[sourceIdx]
            partialSourceProductionForecastAvailable = regionConfig["PARTIAL_FORECAST_AVAILABILITY_LIST"][sourceIdx]
            partialForecastHours =  regionConfig["PARTIAL_FORECAST_HOURS"]
            print(inFileName)
            print(weatherForecastInFileName)
            isRenewableSource = False
            numFeatures = firstTierConfig["NUM_FEATURES"]
            numWeatherFeatures = 0
            if (source == "SOLAR" or source == "WIND" or source == "HYDRO"):
                isRenewableSource = True
            if (isRenewableSource == True):
                numWeatherFeatures = firstTierConfig["NUM_WEATHER_FEATURES"]

            for exptNum in range(NUMBER_OF_EXPERIMENTS):
                outFileName = outFileNamePrefix + "_" + source.lower() + "_iter" + str(exptNum) + ".csv"
                periodRMSE, periodMAPE = [], []
                
                periodIdx = 0
                for period in trainTestPeriodConfig:
                    print(trainTestPeriodConfig[period])
                    datasetLimiter = trainTestPeriodConfig[period]["DATASET_LIMITER"]
                    numTestDays = trainTestPeriodConfig[period]["NUM_TEST_DAYS"]
                    numValDays = firstTierConfig["NUM_VAL_DAYS"]
                    weatherDatasetLimiter = datasetLimiter//24*PREDICTION_WINDOW_HOURS
                    print(numTestDays)

                    print("Initializing...")
                    dataset, dateTime, bufferPeriod, bufferDates, weatherDataset = initialize(
                                inFileName, weatherForecastInFileName, sourceCol,
                                datasetLimiter, weatherDatasetLimiter)
                    # bufferPeriod is for the last test date, if prediction period is beyond 24 hours
                    print("***** Initialization done *****")

                    # split into train and test
                    print("Spliting dataset into train/test...")
                    trainData, valData, testData, fullTrainData = common.splitDataset(dataset.values, numTestDays, 
                                                            numValDays)
                    trainDates = dateTime[: -(numTestDays*24)]
                    fullTrainDates = np.copy(trainDates)
                    trainDates, validationDates = trainDates[: -(numValDays*24)], trainDates[-(numValDays*24):]
                    testDates = dateTime[-(numTestDays*24):]
                    bufferPeriod = bufferPeriod.values
                    trainData = trainData[:, sourceCol: sourceCol+numFeatures]
                    valData = valData[:, sourceCol: sourceCol+numFeatures]
                    testData = testData[:, sourceCol: sourceCol+numFeatures]
                    partialSourceProductionForecast = None
                        
                    bufferPeriod = bufferPeriod[:, sourceCol: sourceCol+numFeatures]
                    if(len(bufferDates)>0):
                        testDates = np.append(testDates, bufferDates)
                        testData = np.vstack((testData, bufferPeriod))

                    print("TrainData shape: ", trainData.shape) # (days x hour) x features
                    print("ValData shape: ", valData.shape) # (days x hour) x features
                    print("TestData shape: ", testData.shape) # (days x hour) x features

                    wTrainData, wValData, wTestData, wFullTrainData = None, None, None, None
                    if (isRenewableSource):
                        wTrainData, wValData, wTestData, wFullTrainData = common.splitWeatherDataset(
                                weatherDataset.values, numTestDays, numValDays, PREDICTION_WINDOW_HOURS)
                        print("WeatherTrainData shape: ", wTrainData.shape) # (days x hour) x features
                        print("WeatherValData shape: ", wValData.shape) # (days x hour) x features
                        print("WeatherTestData shape: ", wTestData.shape) # (days x hour) x features

                    print("***** Dataset split done *****")

                    trainData = fillMissingData(trainData)
                    valData = fillMissingData(valData)
                    testData = fillMissingData(testData)
                    featureList = dataset.columns.values
                    featureList = featureList[sourceCol:sourceCol+numFeatures].tolist()

                    print("Scaling data...")
                    trainData, valData, testData, ftMin, ftMax = common.scaleDataset(trainData, valData, testData)
                    print(trainData.shape, valData.shape, testData.shape)

                    wFtMin = []
                    wFtMax = []
                    if(isRenewableSource):
                        wTrainData = fillMissingData(wTrainData)
                        wValData = fillMissingData(wValData)
                        wTestData = fillMissingData(wTestData)
                        featureList.extend(weatherDataset.columns.values)
                        wTrainData, wValData, wTestData, wFtMin, wFtMax = common.scaleDataset(wTrainData, wValData, wTestData)
                        print(wTrainData.shape, wValData.shape, wTestData.shape)

                    print("Features: ", featureList)
                        
                    if (partialSourceProductionForecastAvailable):
                        print("Partial forecast available for source: ", source)
                        partialSourceProductionForecast = dataset["avg_"+source.lower()+"_production_forecast"].iloc[-numTestDays*24:].values
                        partialSourceProductionForecast = common.scaleColumn(partialSourceProductionForecast, 
                                ftMin[DEPENDENT_VARIABLE_COL], ftMax[DEPENDENT_VARIABLE_COL])
                        # print(partialSourceProductionForecast, ftMax[DEPENDENT_VARIABLE_COL], ftMin[DEPENDENT_VARIABLE_COL])
                    print("***** Data scaling done *****")


                    if (periodIdx == len(trainTestPeriodConfig)-1):
                        print("Saving min & max values for each column in file...")
                        with open(SAVED_MODEL_LOCATION+region+"_"+source+"_min_max_values.txt", "w") as f:
                            f.writelines(str(ftMin))
                            f.write("\n")
                            f.writelines(str(ftMax))
                            f.write("\n")
                            if (isRenewableSource):
                                f.writelines(str(wFtMin))
                                f.write("\n")
                                f.writelines(str(wFtMax))
                                f.write("\n")
                        print("Min-max values saved")

                    ######################## START #####################                    
                    print("Iteration: ", exptNum)
                    bestModel = trainingandValidationPhase(trainData, wTrainData, valData, wValData, 
                                                           firstTierConfig, SAVED_MODEL_LOCATION, region, source)

                    history = valData[-TRAINING_WINDOW_HOURS:, :]
                    weatherData = None
                    if (isRenewableSource):
                        weatherData = wValData[-PREDICTION_WINDOW_HOURS:, :]
                        print("weatherData shape:", weatherData.shape)
                    history = history.tolist()

                    bestRMSE, bestMAPE = [], []
                    predictedData = getDayAheadForecasts(bestModel, history, testData, 
                                        numFeatures+numWeatherFeatures, wTestData, weatherData, partialSourceProductionForecast)
                    print("***** Forecast done *****")
                    
                    unscaledTestData, unscaledPredictedData, formattedTestDates, rmseScore, mapeScore = getUnscaledForecastsAndForecastAccuracy(
                                                                        testData, testDates, predictedData, 
                                                                        ftMin, ftMax)
                    
                    print("[BESTMODEL] Overall RMSE score: ", rmseScore)
                    print("[BESTMODEL] Overall MAPE score: ", mapeScore)
                    # print(scores)
                    bestRMSE.append(rmseScore)
                    bestMAPE.append(mapeScore)

                    
                    print("[BEST] Average RMSE after ", NUMBER_OF_EXPERIMENTS, " expts: ", np.mean(bestRMSE))
                    print("[BEST] Average MAPE after ", NUMBER_OF_EXPERIMENTS, " expts: ", np.mean(bestMAPE))
                    print(bestRMSE)
                    print(bestMAPE)
                    periodRMSE.append(bestRMSE)
                    periodMAPE.append(bestMAPE)

                    writeSourceProductionForecastsToFile(formattedTestDates, unscaledTestData, unscaledPredictedData,
                                                        periodIdx, source, outFileName)
                    periodIdx +=1
                    ######################## END #####################

                ###
                # common.dumpRandomDataToFile("../data/"+region+"/fuel_forecast/"+region+
                #         "_RMSE_iter"+str(exptNum)+source.lower()+".txt", str(periodRMSE), "w")
                # common.dumpRandomDataToFile("../data/"+region+"/fuel_forecast/"+region+
                #         "_MAPE_iter"+str(exptNum)+source.lower()+".txt", str(periodMAPE), "w")
                ###

                print("RMSE: ", periodRMSE)
                print("MAPE: ", periodMAPE)
            sourceIdx += 1

            print("####################", region, source, " done ####################\n\n")
        print("Source production forecast for region: ", region, " done.")
        aggregateDataAndGenerateForecastFile(firstTierConfig, sourceList, weatherForecastInFileName,
                                             outFileNamePrefix, aggregatedForecastFileName, 
                                             isRealTime=False, startDate=None)
    return

def runFirstTierInRealTime(configFileName, regionList, startDate, electricityDataDate, solWindFcstData,
                           realTimeFileDir, realTimeWeatherFileDir, creationTimeInUTC, version):
    global TRAINING_WINDOW_HOURS
    global PREDICTION_WINDOW_HOURS
    global MODEL_SLIDING_WINDOW_LEN
    with open(configFileName, "r") as configFile:
        firstTierConfig = json.load(configFile)
    TRAINING_WINDOW_HOURS = firstTierConfig["TRAINING_WINDOW_HOURS"]
    PREDICTION_WINDOW_HOURS = firstTierConfig["PREDICTION_WINDOW_HOURS"]
    MODEL_SLIDING_WINDOW_LEN = firstTierConfig["MODEL_SLIDING_WINDOW_LEN"]

    aggregatedForecastFileNames = {}
    for region in regionList:
        print("CarbonCast: ANN model for region:", region)
        # Region-specific settings
        regionConfig = firstTierConfig[region]
        sourceList = regionConfig["SOURCES"]
        sourceCols = regionConfig["SOURCE_COL"]
        inFileName = f"{realTimeFileDir}{region}/{region}_{electricityDataDate}.csv"
        weatherFile = f"{realTimeWeatherFileDir}{region}/{region}_weather_forecast_{startDate}.csv"
        SAVED_MODEL_LOCATION = f"{firstTierConfig['SAVED_MODEL_LOCATION']}{region}/"
        output96h = f"{realTimeFileDir}{region}/{region}_96hr_forecasts_{startDate}.csv"
        aggregatedForecastFileNames[region] = output96h

        for idx, source in enumerate(sourceList):
            print(inFileName)
            print(weatherFile)
            colIdx = sourceCols[idx] + 2  # adjust for added creation_time & version columns
            isRenewable = source in ("SOLAR", "WIND", "HYDRO")

            # Initialize and scale data
            dataset, testDates, weatherDataset = initializeInRealTime(
                inFileName, weatherFile, colIdx, isRenewable
            )
            testData = fillMissingData(np.array(dataset.values[:, colIdx:colIdx+1]))
            wData = fillMissingData(np.array(weatherDataset.values))
            ftMin, ftMax, wMin, wMax = common.getMinMaxFeatureValues(
                f"{SAVED_MODEL_LOCATION}{region}_{source}_min_max_values.txt",
                areForecastsFeatures=isRenewable
            )
            testData = common.scaleTestDataWithTrainingValues(testData, ftMin, ftMax)
            wData = common.scaleTestDataWithTrainingValues(wData, wMin, wMax)

            # Handle partial SOLAR/WIND forecasts
            partialForecast = None
            if solWindFcstData is not None and source in ("SOLAR", "WIND"):
                print("Partial forecast available for source:", source)
                col_name = f"avg_{source.lower()}_production_forecast"
                dataset[col_name] = np.nan
                vals = solWindFcstData[col_name].values
                n_vals = len(vals)
                dataset.loc[dataset.index[-n_vals:], col_name] = vals
                last24 = dataset[col_name].iloc[-24:].values
                partialForecast = common.scaleColumn(
                    last24, ftMin[DEPENDENT_VARIABLE_COL], ftMax[DEPENDENT_VARIABLE_COL]
                )

            # Load model
            model = load_model(f"{SAVED_MODEL_LOCATION}/{region}_{source.upper()}_best_model_ann.h5")
            history = testData[-TRAINING_WINDOW_HOURS:].tolist()

            # Determine feature count
            numFeatures = firstTierConfig["NUM_FEATURES"]
            if isRenewable:
                numFeatures += firstTierConfig.get("NUM_WEATHER_FEATURES", 0)

            # Predict
            yhat = getSourceProductionForecastsInRealTime(
                model,
                history,
                testData,
                numFeatures,
                wTestData=wData,
                weatherData=None,
                partialSourceProductionForecast=partialForecast,
                isRenewableSource=isRenewable
            )

            # Inverse scale and write output
            unscaled = common.inverseDataScaling(
                yhat.flatten(), ftMax[DEPENDENT_VARIABLE_COL], ftMin[DEPENDENT_VARIABLE_COL]
            )
            outFile = f"{realTimeFileDir}{region}/fuel_forecast/{region}_ANN_{source.lower()}_{startDate}.csv"
            writeRealTimeSourceProductionForecastsToFile(
                testDates, unscaled, source, outFile, creationTimeInUTC, version
            )
            print(f"#################### {region} {source} done ####################\n")

        # Aggregate across sources
        aggregateDataAndGenerateForecastFile(
            firstTierConfig, sourceList, weatherFile,
            f"{realTimeFileDir}{region}/fuel_forecast/{region}_ANN", output96h,
            isRealTime=True, startDate=startDate
        )
    return aggregatedForecastFileNames



def aggregateDataAndGenerateForecastFile(firstTierConfig, sourceList, weatherForecastFile,
                                         sourceForecastFileNamePrefix, aggregatedForecastFileName,
                                         isRealTime = False, startDate=None):
    
    weatherDatasetStartRow = firstTierConfig["ROW_START_FOR_2020"]
    weatherDatasetEndRow = firstTierConfig["ROW_END_FOR_2022"]
    sourceForecastDatasetEndRow = firstTierConfig["SOURCE_FORECAST_ROW_END_FOR_2022"]

    weatherDataset = pd.read_csv(weatherForecastFile, header=0, index_col=["datetime"])
    if (isRealTime is False):
        weatherDataset = weatherDataset[weatherDatasetStartRow:weatherDatasetEndRow]
    modifiedDataset = weatherDataset.copy()
    for source in sourceList:
        sourceForecastFileName = sourceForecastFileNamePrefix + "_" + source.lower() + "_iter0.csv" # TODO: for now, only 1 iteration. generalize later
        if (isRealTime is True and startDate is not None):
            sourceForecastFileName = sourceForecastFileNamePrefix + "_" + source.lower() + "_"+str(startDate)+".csv"
        sourceForecastDataset = pd.read_csv(sourceForecastFileName, header=0, index_col=["datetime"])
        if (isRealTime is False):
            sourceForecastDataset = sourceForecastDataset[:sourceForecastDatasetEndRow]
        forecastColumnName = "avg_"+source.lower()+"_production_forecast"
        modifiedDataset[forecastColumnName] = sourceForecastDataset[forecastColumnName].values
    print(modifiedDataset.shape)
    # print(modifiedDataset.head(2))
    # print(modifiedDataset.tail(2))

    # print("Writing weather+source production forecasts to file...")
    modifiedDataset.to_csv(aggregatedForecastFileName)
    # print("All forecasts written to a single file")
    return

def initialize(inFileName, weatherForecastInFileName, startCol, datasetLimiter,
                weatherDatasetLimiter):

    global BUFFER_HOURS
    # load the new file
    dataset = pd.read_csv(inFileName, header=0, infer_datetime_format=True, 
                            parse_dates=['UTC time'], index_col=['UTC time'])

    # print(dataset.head())
    # print(dataset.columns)
    dateTime = dataset.index.values

    # weatherDataset = pd.read_csv(weatherForecastInFileName, header=0, infer_datetime_format=True, 
    #                         parse_dates=['UTC time'], index_col=['UTC time'])
    weatherDataset = pd.read_csv(weatherForecastInFileName, header=0, infer_datetime_format=True, 
                            parse_dates=['datetime'], index_col=['datetime'])
    # print(weatherDataset.head())
    
    print("\nAdding features related to date & time...")
    modifiedDataset = common.addDateTimeFeatures(dataset, dateTime, startCol)
    dataset = modifiedDataset
    print("Features related to date & time added")

    bufferPeriod = dataset[datasetLimiter:datasetLimiter+BUFFER_HOURS]
    dataset = dataset[:datasetLimiter]
    bufferDates = dateTime[datasetLimiter:datasetLimiter+BUFFER_HOURS]
    dateTime = dateTime[:datasetLimiter]

    weatherDataset = weatherDataset[:weatherDatasetLimiter]
    
    for i in range(startCol, len(dataset.columns.values)):
        col = dataset.columns.values[i]
        dataset[col] = dataset[col].astype(np.float64)
        # print(col, dataset[col].dtype)

    # print("Getting contribution of each energy source...")
    # contribution = getAvgContributionBySource(dataset)
    # print(contribution)

    return dataset, dateTime, bufferPeriod, bufferDates, weatherDataset

def initializeInRealTime(inFileName, weatherForecastInFileName, startCol, isRenewableSource):
    # load the new file
    dataset = pd.read_csv(inFileName, header=0, infer_datetime_format=True, 
                            parse_dates=['UTC time'], index_col=['UTC time'])

    weatherDataset = pd.read_csv(weatherForecastInFileName, header=0, infer_datetime_format=True, 
                            parse_dates=['datetime'], index_col=['datetime'])
    weatherDateTime = weatherDataset.index.values
    # Adding in weather dataset, as we need for 96 hours
    weatherDataset = weatherDataset.iloc[:, 2:] # this is because we have now added creation time & version
    modifiedWeatherDataset = common.addDateTimeFeatures(weatherDataset, weatherDateTime, -1)
    if (isRenewableSource is True):
        weatherDataset = modifiedWeatherDataset
    else:
        weatherDataset = modifiedWeatherDataset.iloc[:, :5]
    print("Features related to date & time added")
    
    for i in range(startCol, len(dataset.columns.values)):
        col = dataset.columns.values[i]
        dataset[col] = dataset[col].astype(np.float64)

    return dataset, weatherDateTime, weatherDataset

def fillMissingData(data): # If some data is missing (NaN), use the same value as that of the previous row
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if(np.isnan(data[i, j])):
                data[i, j] = data[i-1, j]
    return data

def trainingandValidationPhase(trainData, wTrainData, valData, wValData, 
                               firstTierConfig, savedModelLocation, region, source):
    global TRAINING_WINDOW_HOURS
    print("\nManipulating training data...")
    X, y = manipulateTrainingDataShape(trainData, TRAINING_WINDOW_HOURS, wTrainData)
    print("\nManipulating validation data...")
    # Next line actually labels validation data
    valX, valY = manipulateTrainingDataShape(valData, TRAINING_WINDOW_HOURS, wValData)
                    
    print("***** Training and validation data manipulation done *****")
    print("X.shape, y.shape: ", X.shape, y.shape)
    hyperParams = getANNHyperParams(firstTierConfig)                
    print("\n[BESTMODEL] Starting training...")
    bestTrainedModel = trainANN(X, y, valX, valY, hyperParams, savedModelLocation, region, source)
    print("***** Training done *****")
    return bestTrainedModel

# convert training data into inputs and outputs (labels)
def manipulateTrainingDataShape(data, labelWindowHours, weatherData = None):
    global TRAINING_WINDOW_HOURS
    global PREDICTION_WINDOW_HOURS

    print("Data shape: ", data.shape)
    global PREDICTION_WINDOW_HOURS
    X, y, weatherX = list(), list(), list()
    weatherIdx = 0
    hourIdx = 0
    # step over the entire history one time step at a time
    for i in range(len(data)-(TRAINING_WINDOW_HOURS+labelWindowHours)+1):
        # define the end of the input sequence
        trainWindow = i + TRAINING_WINDOW_HOURS
        labelWindow = trainWindow + labelWindowHours
        xInput = data[i:trainWindow, :]
        # xInput = xInput.reshape((len(xInput), 1))
        X.append(xInput)
        if(weatherData is not None):
            weatherX.append(weatherData[weatherIdx:weatherIdx+TRAINING_WINDOW_HOURS])
            weatherIdx +=1
            hourIdx +=1
            if(hourIdx ==24):
                hourIdx = 0
                weatherIdx += (PREDICTION_WINDOW_HOURS-24)
        y.append(data[trainWindow:labelWindow, DEPENDENT_VARIABLE_COL])
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    if(weatherData is not None):
        weatherX = np.array(weatherX, dtype=np.float64)
        X = np.append(X, weatherX, axis=2)
    return X, y

def manipulateTestDataShape(data, isDates=False):
    global MODEL_SLIDING_WINDOW_LEN
    global PREDICTION_WINDOW_HOURS
    X = list()
    # step over the entire history one time step at a time
    for i in range(0, len(data)-(PREDICTION_WINDOW_HOURS)+1, MODEL_SLIDING_WINDOW_LEN):
        # define the end of the input sequence
        predictionWindow = i + PREDICTION_WINDOW_HOURS
        X.append(data[i:predictionWindow])
    if (isDates is False):
        X = np.array(X, dtype=np.float64)
    else:
        X = np.array(X)
    return X


def trainANN(trainX, trainY, valX, valY, hyperParams, savedModelLocation, region, source):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    epochs = hyperParams["epoch"]
    batchSize = hyperParams["batchsize"]
    lossFunc = hyperParams["loss"]
    actvFunc = hyperParams["actv"]
    hiddenDims = hyperParams["hidden"]
    learningRates = hyperParams["lr"]
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(hiddenDims[0], input_shape=(n_timesteps, n_features), activation=actvFunc)) # 50
    model.add(Dense(hiddenDims[1], activation=actvFunc)) # 34
    model.add(Dense(n_outputs))
    
    opt = tf.keras.optimizers.Adam(learning_rate = learningRates)
    model.compile(loss=lossFunc, optimizer=opt,
                    metrics=["mean_absolute_error"])
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
    mc = ModelCheckpoint(savedModelLocation+region+"_"+source+"_best_model_ann.h5", monitor="val_loss", mode="min", verbose=1, save_best_only=True)
    # fit network
    # hist = model.fit(trainX, trainY, epochs=epochs, batch_size=bSize, verbose=verbose)
    hist = model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize[0], verbose=2,
                        validation_data=(valX, valY), callbacks=[es, mc])
    model = load_model(savedModelLocation+region+"_"+source+"_best_model_ann.h5")
    common.showModelSummary(hist, model)
    print("Number of features used in training: ", n_features)
    return model

def getDayAheadForecasts(model, history, testData, 
                            numFeatures,
                            wTestData = None, weatherData = None, 
                            partialSourceProductionForecast = None):
    global TRAINING_WINDOW_HOURS
    global MODEL_SLIDING_WINDOW_LEN
    global PREDICTION_WINDOW_HOURS
    global BUFFER_HOURS
    # walk-forward validation over each day
    print("Testing (day ahead forecasts)...")
    predictions = list()
    weatherIdx = 0
    for i in range(0, ((len(testData)//24)-(BUFFER_HOURS//24))):
        dayAheadPredictions = list()
        # predict n days, 1 day at a time
        tempHistory = history.copy()
        currentDayHours = i* MODEL_SLIDING_WINDOW_LEN
        for j in range(0, PREDICTION_WINDOW_HOURS, 24):
            if (weatherData is not None):
                yhat_sequence, newTrainingData = getForecasts(model, tempHistory, 
                            numFeatures, weatherData[j:j+24])
            else:
                yhat_sequence, newTrainingData = getForecasts(model, tempHistory, 
                            numFeatures, None)
            # add current prediction to history for predicting the next day
            if (j==0 and partialSourceProductionForecast is not None):
                for k in range(24):
                    yhat_sequence[k] = partialSourceProductionForecast[currentDayHours+k]
            dayAheadPredictions.extend(yhat_sequence)
            latestHistory = testData[currentDayHours+j:currentDayHours+j+24, :].tolist()
            for k in range(24):
                latestHistory[k][DEPENDENT_VARIABLE_COL] = yhat_sequence[k]
            tempHistory.extend(latestHistory)

        # get real observation and add to history for predicting the next day
        
        history.extend(testData[currentDayHours:currentDayHours+MODEL_SLIDING_WINDOW_LEN, :].tolist())
        predictions.append(dayAheadPredictions)
        if (wTestData is not None):
            weatherData = wTestData[weatherIdx:weatherIdx+PREDICTION_WINDOW_HOURS, :]
            weatherIdx +=PREDICTION_WINDOW_HOURS

    # evaluate predictions days for each day
    predictedData = np.array(predictions, dtype=np.float64)
    return predictedData

def getSourceProductionForecastsInRealTime(model, history, testData, 
                            numFeatures,
                            wTestData = None, weatherData = None, 
                            partialSourceProductionForecast = None,
                            isRenewableSource=False):
    # walk-forward validation over each day
    if weatherData is None:
        if wTestData is not None:
            weatherData = wTestData
        else:
            # No weather at all: make an empty 2D array with correct rows
            weatherData = np.empty((len(testData), 0), dtype=np.float64)

    print("Testing...")
    predictions = []
    weatherIdx = 0

    # How many full days of testData we have
    numDays = len(testData) // 24

    for day in range(numDays):
        dayAheadPredictions = []
        tempHistory = history.copy()
        baseHour = day * MODEL_SLIDING_WINDOW_LEN

        # Within each day, we step in 24-hour blocks up to PREDICTION_WINDOW_HOURS
        for offset in range(0, PREDICTION_WINDOW_HOURS, 24):
            if isRenewableSource:
                window = weatherData[offset:offset + 24]
                yhat_seq = getForecastsInRealTime(model,
                                                  tempHistory,
                                                  numFeatures,
                                                  window)
            else:
                # non-renewables only look at the first 5 weather features
                window = weatherData[offset:offset + 24, :5]
                yhat_seq = getForecastsInRealTime(model,
                                                  tempHistory,
                                                  numFeatures,
                                                  window)

            # For the first 24h block, override with partial forecasts if given
            if day == 0 and offset == 0 and partialSourceProductionForecast is not None:
                for h in range(24):
                    # on day 0, baseHour == 0, so we can just index [h]
                    yhat_seq[h] = partialSourceProductionForecast[h]
                    dayAheadPredictions.extend(yhat_seq)

                    # Roll the history window forward by injecting this block’s outputs
                    for h in range(24):
                        tempHistory[h] = yhat_seq[h]

        # After forecasting this day, add the next real observation slice into history
        start_obs = baseHour
        end_obs = baseHour + MODEL_SLIDING_WINDOW_LEN
        history.extend(testData[start_obs:end_obs, :].tolist())
        predictions.append(dayAheadPredictions)

        # Advance the weatherData pointer for the next day
        if wTestData is not None:
            weatherData = wTestData[weatherIdx:weatherIdx + PREDICTION_WINDOW_HOURS]
            weatherIdx += PREDICTION_WINDOW_HOURS

    # Finally, convert list-of-lists to a 2D NumPy array
    target_len = PREDICTION_WINDOW_HOURS
    cleaned = []
    for day_preds in predictions:
        if len(day_preds) < target_len:
            day_preds = day_preds + [np.nan] * (target_len - len(day_preds))
        elif len(day_preds) > target_len:
            day_preds = day_preds[:target_len]
        cleaned.append(day_preds)

    predictedData = np.array(cleaned, dtype=np.float64)
    return predictedData
    # print("Testing...")
    # predictions = list()
    # weatherIdx = 0
    # for i in range(0, ((len(testData)//24))):
    #     dayAheadPredictions = list()
    #     tempHistory = history.copy()
    #     currentDayHours = i* MODEL_SLIDING_WINDOW_LEN
    #     for j in range(0, PREDICTION_WINDOW_HOURS, 24):
    #         if (isRenewableSource is True):
    #             yhat_sequence = getForecastsInRealTime(model, tempHistory, 
    #                         numFeatures, weatherData[j:j+24])
    #         else:
    #             yhat_sequence = getForecastsInRealTime(model, tempHistory, 
    #                         numFeatures, weatherData[j:j+24, :5])
    #         # add current prediction to history for predicting the next day
    #         if (j==0 and partialSourceProductionForecast is not None):
    #             for k in range(24):
    #                 yhat_sequence[k] = partialSourceProductionForecast[currentDayHours+k]
    #         dayAheadPredictions.extend(yhat_sequence)
    #         for k in range(24):
    #             tempHistory[k] = yhat_sequence[k]
    #     # get real observation and add to history for predicting the next day
    #     history.extend(testData[currentDayHours:currentDayHours+MODEL_SLIDING_WINDOW_LEN, :].tolist())
    #     predictions.append(dayAheadPredictions)
    #     if (wTestData is not None):
    #         weatherData = wTestData[weatherIdx:weatherIdx+PREDICTION_WINDOW_HOURS, :]
    #         weatherIdx +=PREDICTION_WINDOW_HOURS

    # # evaluate predictions days for each day
    # predictedData = np.array(predictions, dtype=np.float64)
    # return predictedData


def getForecasts(model, history, numFeatures, weatherData):
    global TRAINING_WINDOW_HOURS

    #debugging
    # for i, h in enumerate(history[:10]):
    #     print(f"{i:2d} – type: {type(h)}, ",
    #         "len:" if hasattr(h, "__len__") else "",
    #         getattr(h, "__len__",  lambda: None)())

    # ——— SANITY-CHECK & UNIFORMIZE history ———
    processed = []
    for i, item in enumerate(history):
        if isinstance(item, (pd.DataFrame, pd.Series)):
            arr = item.values.astype(np.float64)
        else:
            arr = np.asarray(item, dtype=np.float64)
        if arr.ndim != 1:
            arr = arr.ravel()
        processed.append(arr)

    # ensure they’re all the same length
    lengths = [a.shape[0] for a in processed]
    if len(set(lengths)) != 1:
        raise ValueError(f"history entries have mismatched lengths: {lengths}")
    
    # now stack into a 2D array
    data = np.vstack(processed)  # shape = (n_steps, n_features)


    # flatten data
    # data = np.array(history, dtype=np.float64)
      
    # if isinstance(history, (pd.DataFrame, pd.Series)):
    #     # Convert DataFrame to numpy array properly
    #     data = history.values.astype(np.float64)
    # elif isinstance(history, list):
    #     # If it's a list of DataFrames or mixed objects
    #     processed_history = []
    #     for item in history:
    #         if isinstance(item, (pd.DataFrame, pd.Series)):
    #             # Extract numeric values from each DataFrame
    #             processed_history.append(item.values.astype(np.float64))
    #         elif isinstance(item, (list, tuple, np.ndarray)):
    #             # If it's already a numeric sequence
    #             processed_history.append(np.array(item, dtype=np.float64))
    #         else:
    #             # Skip or handle non-compatible items
    #             print(f"Skipping incompatible item type: {type(item)}")
        
    #     # Now ensure all arrays have the same shape
    #     # Find the most common shape
    #     shapes = [arr.shape for arr in processed_history if hasattr(arr, 'shape')]
    #     if not shapes:
    #         raise ValueError("No valid data found in history")
            
    #     # Convert all arrays to the same shape (if needed)
    #     uniform_history = []
    #     for arr in processed_history:
    #         if hasattr(arr, 'shape') and arr.shape == shapes[0]:
    #             uniform_history.append(arr)
        
    #     # Now convert to a numpy array
    #     data = np.array(uniform_history, dtype=np.float64)
    # else:
    #     # Try direct conversion for other types
    #     data = np.array(history, dtype=np.float64)
    # data = np.stack([
    # h.to_numpy().reshape(-1) if isinstance(h, pd.DataFrame)
    #     else (h.to_numpy() if isinstance(h, pd.Series)
    #         else np.array(h).reshape(-1))
    #     for h in history
    # ], axis=0).astype(np.float64)
    # retrieve last observations for input data
    input_x = data[-TRAINING_WINDOW_HOURS:]
    if (weatherData is not None):
        input_x = np.append(input_x, weatherData, axis=1)
    # reshape into [1, n_input, num_features]
    input_x = input_x.reshape((1, len(input_x), numFeatures))
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat, input_x

def getForecastsInRealTime(model, history, numFeatures, weatherData):
    global TRAINING_WINDOW_HOURS
    # flatten data
    # data = np.array(history, dtype=np.float64)
    # data = np.reshape(data, (data.shape[0], 1))
    # # retrieve last observations for input data
    # input_x = data[-TRAINING_WINDOW_HOURS:]
    # if (weatherData is not None):
    #     input_x = np.append(input_x, weatherData, axis=1)
        # ——— SANITY-CHECK & UNIFORMIZE history ———

    # # now stack into a 2D array
    # data = np.vstack(processed)  # shape = (n_steps, n_features)

    # # reshape into [1, n_input, num_features]
    # input_x = input_x.reshape((1, len(input_x), numFeatures))
    # yhat = model.predict(input_x, verbose=0)
    # yhat = yhat[0]
    # return yhat

    # ——— 1) Turn history into a uniform 2D float array ———

    # now all arr.shape == (L,), so shape[0] is always valid
    processed = [
        np.asarray(item, dtype=np.float64).ravel()
        for item in history
    ]
    # All entries must now be 1-D, so shape[0] always exists
    lengths = [arr.shape[0] for arr in processed]
    if len(set(lengths)) != 1:
        raise ValueError(f"history entries have mismatched lengths: {lengths}")
    # Stack into a (n_steps × feature_dim) array
    data = np.vstack(processed)

    
    # ——— 2) Slice out the last TRAINING_WINDOW_HOURS rows ———
    input_x = data[-TRAINING_WINDOW_HOURS:, :]

    # ——— 3) If there’s weather data, append it as new columns ———
    if weatherData is not None:
        # make sure weatherData is a 2D array of shape (TRAINING_WINDOW_HOURS, n_weather_feats)
        input_x = np.concatenate([input_x, weatherData], axis=1)

    # ——— 4) Finally reshape to (1, time_steps, total_features) ———
    input_x = input_x.reshape((1, input_x.shape[0], numFeatures))

    # ——— 5) Predict and return only the forecast vector ———
    yhat = model.predict(input_x, verbose=0)[0]
    return yhat

def getANNHyperParams(firstTierConfig):
    hyperParams = {}
    modelHyperparamsFromConfigFile = firstTierConfig["FIRST_TIER_ANN_MODEL_HYPERPARAMS"]
    hyperParams["epoch"] = modelHyperparamsFromConfigFile["EPOCH"]
    hyperParams["batchsize"] = modelHyperparamsFromConfigFile["BATCH_SIZE"]
    hyperParams["actv"] = modelHyperparamsFromConfigFile["ACTIVATION_FUNC"]
    hyperParams["loss"] = modelHyperparamsFromConfigFile["LOSS_FUNC"]
    hyperParams["lr"] = modelHyperparamsFromConfigFile["LEARNING_RATE"]
    hyperParams["hidden"] = modelHyperparamsFromConfigFile["HIDDEN_UNITS"]
    return hyperParams

def getUnscaledForecastsAndForecastAccuracy(testData, testDates, predictedData, ftMin, ftMax):
    global MODEL_SLIDING_WINDOW_LEN
    global PREDICTION_WINDOW_HOURS
    actualData = manipulateTestDataShape(testData[:, DEPENDENT_VARIABLE_COL], False)
    formattedTestDates = manipulateTestDataShape(testDates, True)
    formattedTestDates = np.reshape(formattedTestDates, 
            formattedTestDates.shape[0]*formattedTestDates.shape[1])
    actualData = actualData.astype(np.float64)
    print("ActualData shape: ", actualData.shape)
    actual = np.reshape(actualData, actualData.shape[0]*actualData.shape[1])
    print("actual.shape: ", actual.shape)
    unscaledTestData = common.inverseDataScaling(actual, ftMax[DEPENDENT_VARIABLE_COL], 
                        ftMin[DEPENDENT_VARIABLE_COL])
    predictedData = predictedData.astype(np.float64)
    print("PredictedData shape: ", predictedData.shape)
    predicted = np.reshape(predictedData, predictedData.shape[0]*predictedData.shape[1])
    print("predicted.shape: ", predicted.shape)
    unscaledPredictedData = common.inverseDataScaling(predicted, 
                ftMax[DEPENDENT_VARIABLE_COL], ftMin[DEPENDENT_VARIABLE_COL])
    rmseScore, mapeScore = common.getScores(actualData, predictedData, 
                                unscaledTestData, unscaledPredictedData)    
    return unscaledTestData, unscaledPredictedData, formattedTestDates, rmseScore, mapeScore

def writeSourceProductionForecastsToFile(formattedTestDates, unscaledTestData, unscaledPredictedData,
                                        period, source, outFileName):
    data = []
    
    for i in range(len(unscaledPredictedData)):
        row = []
        row.append(str(formattedTestDates[i]))
        row.append(str(unscaledTestData[i]))
        row.append(str(unscaledPredictedData[i]))
        data.append(row)
    writeMode = "w"
    if (period > 0):
        writeMode = "a"
    common.writeOutFile(outFileName, data, source.lower(), writeMode)
    return

def writeRealTimeSourceProductionForecastsToFile(formattedTestDates, unscaledPredictedData,
                                        source, outFileName, creationTimeInUTC, version):
    data = []
    for date, forecast in zip(formattedTestDates, unscaledPredictedData):
        data.append([str(date), creationTimeInUTC, version, str(forecast)])
    if len(unscaledPredictedData) > len(formattedTestDates):
        extra = len(unscaledPredictedData) - len(formattedTestDates)
        print(f"Warning: {extra} extra forecast entries without corresponding dates; ignoring extras.")
    print("Writing to", outFileName, "...")
    fields = [
        "datetime", "creation_time (UTC)", "version",
        f"avg_{source.lower()}_production_forecast"
    ]
    with open(outFileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)
        writer.writerows(data)
    return
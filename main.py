####################################################################################
#   Implementation of the following paper: https://arxiv.org/pdf/1703.07015.pdf    #
#                                                                                  #
#    Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks     #
####################################################################################

# This must be set in the beggining because in model_util, we import it
logger_name = "lstnet"

# Path appended in order to import from util
import sys
sys.path.append('..')
from util.model_util import LoadModel, SaveModel, SaveResults, SaveHistory
from util.Msglog import LogInit

from datetime import datetime

from lstnet_util import GetArguments, LSTNetInit
from lstnet_datautil import DataUtil
from lstnet_model import PreSkipTrans, PostSkipTrans, PreARTrans, PostARTrans, LSTNetModel, ModelCompile
from lstnet_plot import AutoCorrelationPlot, PlotHistory, PlotPrediction

import tensorflow as tf


custom_objects = {
        'PreSkipTrans': PreSkipTrans,
        'PostSkipTrans': PostSkipTrans,
        'PreARTrans': PreARTrans,
        'PostARTrans': PostARTrans
        }

def train(model, data, init, tensorboard = None):
    if init.validate == True:
        val_data = (data.valid[0], data.valid[1])
    else:
        val_data = None

    start_time = datetime.now()
    history = model.fit(
                x = data.train[0],
                y = data.train[1],
                epochs = init.epochs,
                batch_size = init.batchsize,
                validation_data = val_data,
                callbacks = [tensorboard] if tensorboard else None
            )
    end_time = datetime.now()
    log.info("Training time took: %s", str(end_time - start_time))

    return history


if __name__ == '__main__':
    try:
        args = GetArguments()
    except SystemExit as err:
        print("Error reading arguments")
        exit(0)

    test_result = None

    # Initialise parameters
    lstnet_init = LSTNetInit(args)
    
    # Initialise logging
    log = LogInit(logger_name, lstnet_init.logfilename, lstnet_init.debuglevel, lstnet_init.log)
    log.info("Python version: %s", sys.version)
    log.info("Tensorflow version: %s", tf.__version__)
    log.info("Keras version: %s ... Using tensorflow embedded keras", tf.keras.__version__)

    # Dumping configuration
    lstnet_init.dump()

    # Reading data
    Data = DataUtil(lstnet_init.data,
                    lstnet_init.trainpercent,
                    lstnet_init.validpercent,
                    lstnet_init.horizon,
                    lstnet_init.window,
                    lstnet_init.normalise)
    
    # If file does not exist, then Data will not have attribute 'data'
    if hasattr(Data, 'data') is False:
        log.critical("Could not load data!! Exiting")
        exit(1)

    log.info("Training shape: X:%s Y:%s", str(Data.train[0].shape), str(Data.train[1].shape))
    log.info("Validation shape: X:%s Y:%s", str(Data.valid[0].shape), str(Data.valid[1].shape))
    log.info("Testing shape: X:%s Y:%s", str(Data.test[0].shape), str(Data.test[1].shape))

    if lstnet_init.plot == True and lstnet_init.autocorrelation is not None:
        AutoCorrelationPlot(Data, lstnet_init)

    # If --load is set, load model from file, otherwise create model
    if lstnet_init.load is not None:
        log.info("Load model from %s", lstnet_init.load)
        lstnet = LoadModel(lstnet_init.load, custom_objects)
    else:
        log.info("Creating model")
        lstnet = LSTNetModel(lstnet_init, Data.train[0].shape)

    if lstnet is None:
        log.critical("Model could not be loaded or created ... exiting!!")
        exit(1)

    # Compile model
    lstnet_tensorboard = ModelCompile(lstnet, lstnet_init)
    if lstnet_tensorboard is not None:
        log.info("Model compiled ... Open tensorboard in order to visualise it!")
    else:
        log.info("Model compiled ... No tensorboard visualisation is available")

    # Model Training
    if lstnet_init.train is True:
        # Train the model
        log.info("Training model ... ")
        h = train(lstnet, Data, lstnet_init, lstnet_tensorboard)

        # Plot training metrics
        if lstnet_init.plot is True:
            PlotHistory(h.history, ['loss', 'rse', 'corr'], lstnet_init)

        # Saving model if lstnet_init.save is not None.
        # There's no reason to save a model if lstnet_init.train == False
        SaveModel(lstnet, lstnet_init.save)
        if lstnet_init.saveresults == True:
            SaveResults(lstnet, lstnet_init, h.history, test_result, ['loss', 'rse', 'corr'])
        if lstnet_init.savehistory == True:
            SaveHistory(lstnet_init.save, h.history)

    # Validation
    if lstnet_init.train is False and lstnet_init.validate is True:
        loss, rse, corr = lstnet.evaluate(Data.valid[0], Data.valid[1])
        log.info("Validation on the validation set returned: Loss:%f, RSE:%f, Correlation:%f", loss, rse, corr)
    elif lstnet_init.validate == True:
        log.info("Validation on the validation set returned: Loss:%f, RSE:%f, Correlation:%f",
                 h.history['val_loss'][-1], h.history['val_rse'][-1], h.history['val_corr'][-1])

    # Testing evaluation
    if lstnet_init.evaltest is True:
        loss, rse, corr = lstnet.evaluate(Data.test[0], Data.test[1])
        log.info("Validation on the test set returned: Loss:%f, RSE:%f, Correlation:%f", loss, rse, corr)
        test_result = {'loss': loss, 'rse': rse, 'corr': corr}

    # Prediction
    if lstnet_init.predict is not None:
        if lstnet_init.predict == 'trainingdata' or lstnet_init.predict == 'all':
            log.info("Predict training data")
            trainPredict = lstnet.predict(Data.train[0])
        else:
            trainPredict = None
        if lstnet_init.predict == 'validationdata' or lstnet_init.predict == 'all':
            log.info("Predict validation data")
            validPredict = lstnet.predict(Data.valid[0])
        else:
            validPredict = None
        if lstnet_init.predict == 'testingdata' or lstnet_init.predict == 'all':
            log.info("Predict testing data")
            testPredict = lstnet.predict(Data.test[0])
        else:
            testPredict = None

        if lstnet_init.plot is True:
            PlotPrediction(Data, lstnet_init, trainPredict, validPredict, testPredict)

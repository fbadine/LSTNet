import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import autocorrelation_plot

# logging
from __main__ import logger_name
import logging
log = logging.getLogger(logger_name)


def AutoCorrelationPlot(Data, init):
    if Data is not None and init is not None:
        log.info("Plotting autocorrelation ...")
        #
        # init.autocorrelation has the following format: number_of_series,start,end
        # which means that we will be plotting an autocorrelation for number_of_series random series from start to end
        #
        # Here we are transforming this series into a list of integers if possible
        #
        s = [int(i) if i.isdigit() else i for i in init.autocorrelation.split(',')]

        #
        # Check if the first element in the list is an integer
        # and is between 1 and the number of timeseries otherwise set it to the number of timeseries i.e. plot all
        #
        try:
            assert(s[0] and type(s[0]) == int and s[0] > 0 and s[0] <= Data.m)
            number = s[0]
        except AssertionError as err:
            log.warning("The number of series to plot autocorrelation for must be in the range [1,%d]. Setting it to %d", Data.m, Data.m)
            number = Data.m

        #
        # Check if the second element in the list exists (len(s)>1) and is an integer
        # and is less than the length of the timeseries otherwise set it to 0 (start of the timeseries)
        #
        try:
            assert(len(s) > 1 and s[1] and type(s[1]) == int and s[1] < Data.n)
            start_plot = s[1]
        except AssertionError as err:
            log.warning("start must be an integer less than %d. Setting it to 0", Data.n)
            start_plot = 0

        #
        # Check if the third element in the list exists (len(s)>2) and is an integer and is bigger than the start_plot
        # and is less than the length of the timeseries otherwise set it to end of the timeseries
        #
        try:
            assert(len(s) > 2 and s[2] and type(s[2]) == int and s[2] > start_plot and s[2] < Data.n)
            end_plot = s[2]
        except AssertionError as err:
            log.warning("end must be an integer in the range ]%d,%d[. Setting it to %d", start_plot, Data.n, Data.n - 1)
            end_plot = Data.n - 1

        fig = plt.figure()

        log.debug("Plotting autocorrelation for %d random timeseries out of %d. Timeslot from %d to %d", number, Data.m, start_plot, end_plot)
        series = np.random.choice(range(Data.m), number, replace=False)
        for i in series:
            autocorrelation_plot(Data.data[start_plot:end_plot,i])

        fig.canvas.set_window_title('Auto Correlation')
        plt.show()

        if init.save_plot is not None:
            log.debug("Saving autocorrelation plot to: %s", init.save_plot + "_autocorrelation.png")
            fig.savefig(init.save_plot + "_autocorrelation.png")


def PlotHistory(history, metrics, init):
    if history is not None and metrics is not None and init is not None:
        log.info("Plotting history ...")

        # Number of keys present in the history dictionary
        i = 1

        #
        # Number of metrics that were trained and are available in history
        # This will help us determine the width of the canvas as well as correctly set
        # the parameters to subplot
        #
        n = len(history)

        #
        # The number of rows is set so that the training results are on one line and the
        # validation ones are on the second. Therefore:
        #         number of available metrics in history
        # rows = --------------------------------------- = 2 in case of validate=True. Otherwise 1
        #         number of metrics
        #
        rows = int(n / len(metrics))

        #
        # Number of columns i.e. number of different metrics plotted for each of the training and validation
        #
        cols = int(n / rows)

        #
        # Set the plotting image size
        # If the number of columns is greater than 2, choose 16, otherwise 12
        # If the number of rows is greater than 1, choose 10, otherwise 5
        #
        fig = plt.figure(figsize=(16 if cols > 2 else 12, 10 if rows > 1 else 5))

        # Training data history plot
        for m in metrics:
            key = m
            log.debug("Plotting metrics %s", key)
            plt.subplot(rows, cols, i)
            plt.plot(history[key])
            plt.ylabel(m.title())
            plt.xlabel("Epochs")
            plt.title("Training " + m.title())
            i = i + 1

        # Validation data history plot (if available)
        for m in metrics:
            key = "val_" + m
            log.debug("Plotting metrics %s", key)
            # if key is not in history.keys() => --validate was set to False and therefore history for validation is not available
            if key in history.keys():
                plt.subplot(rows, cols, i)
                plt.plot(history[key])
                plt.ylabel(m.title())
                plt.xlabel("Epochs")
                plt.title("Validation " + m.title())
                i = i + 1

        fig.canvas.set_window_title('Training History')
        plt.show()

        if init.save_plot is not None:
            log.debug("Saving training history plot to: %s", init.save_plot + "_training.png")
            fig.savefig(init.save_plot + "_training.png")

def PlotPrediction(Data, init, trainPredict, validPredict, testPredict):
    if Data is not None and init is not None:
        log.info("Plotting Prediction ...")
        #
        # init.series_to_plot has the following format: series_number,start,end
        # which means that we will be plotting series # series_number from start to end
        #
        # Here we are transforming this series into a list of integers if possible
        #
        s = [int(i) if i.isdigit() else i for i in init.series_to_plot.split(',')]

        #
        # Check if the first element in the list is an integer
        # and is less than the number of timeseries otherwise set it to 0
        #
        try:
            assert(s[0] and type(s[0]) == int and s[0] < Data.m)
            series = s[0]
        except AssertionError as err:
            log.warning("The series to plot must be an integer in the range [0,%d[. Setting it to 0", Data.m)
            series = 0

        #
        # Check if the second element in the list exists (len(s)>1) and is an integer
        # and is less than the length of the timeseries otherwise set it to 0 (start of the timeseries)
        #
        try:
            assert(len(s) > 1 and s[1] and type(s[1]) == int and s[1] < Data.n)
            start_plot = s[1]
        except AssertionError as err:
            log.warning("start must be an integer less than %d. Setting it to 0", Data.n)
            start_plot = 0

        #
        # Check if the third element in the list exists (len(s)>2) and is an integer and is bigger than the start_plot
        # and is less than the length of the timeseries otherwise set it to end of the timeseries
        #
        try:
            assert(len(s) > 2 and s[2] and type(s[2]) == int and s[2] > start_plot and s[2] < Data.n)
            end_plot = s[2]
        except AssertionError as err:
            log.warning("end must be an integer in the range ]%d,%d[. Setting it to %d", start_plot, Data.n, Data.n - 1)
            end_plot = Data.n - 1


        #
        # Create empty series of the same length of the data and set the values to nan
        # This way, we can fill the appropriate section for train, valid, test so that 
        # when we print them, they appear at the appropriate loction with respect to the original timeseries
        #
        log.debug("Initialising trainPredictPlot, ValidPredictPlot, testPredictPlot")
        trainPredictPlot      = np.empty((Data.n, Data.m))
        trainPredictPlot[:,:] = np.nan
        validPredictPlot      = np.empty((Data.n, Data.m))
        validPredictPlot[:,:] = np.nan
        testPredictPlot       = np.empty((Data.n, Data.m))
        testPredictPlot[:,:]  = np.nan

        #
        # We use window data to predict a value at horizon from the end of the window, therefore start is
        # is at the end of the horizon
        #
        start = init.window + init.horizon - 1
        end   = start + len(Data.train[0]) # Same length as trainPredict however we might not have trainPredict
        if trainPredict is not None:
            log.debug("Filling trainPredictPlot from %d to %d", start, end)
            trainPredictPlot[start:end, :] = trainPredict

        start = end
        end   = start + len(Data.valid[0]) # Same length as validPredict however we might not have validPredict
        if validPredict is not None:
            log.debug("Filling validPredictPlot from %d to %d", start, end)
            validPredictPlot[start:end, :] = validPredict

        start = end
        end   = start + len(Data.test[0]) # Same length as testPredict however we might not have testPredict
        if testPredict is not None:
            log.debug("Filling testPredictPlot from %d to %d", start, end)
            testPredictPlot[start:end, :] = testPredict

        # Plotting the original series and whatever is available of trainPredictPlot, validPredictPlot and testPredictPlot
        fig = plt.figure()

        plt.plot(Data.data[start_plot:end_plot, series])
        plt.plot(trainPredictPlot[start_plot:end_plot, series])
        plt.plot(validPredictPlot[start_plot:end_plot, series])
        plt.plot(testPredictPlot[start_plot:end_plot, series])

        plt.ylabel("Timeseries")
        plt.xlabel("Time")
        plt.title("Prediction Plotting for timeseries # %d" % (series))
        
        fig.canvas.set_window_title('Prediction')

        plt.show()

        if init.save_plot is not None:
            log.debug("Saving prediction plot to: %s", init.save_plot + "_prediction.png")
            fig.savefig(init.save_plot + "_prediction.png")

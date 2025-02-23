// Accord Audio Library
// The Accord.NET Framework
// http://accord-framework.net
//
// Copyright © César Souza, 2009-2017
// cesarsouza at gmail.com
//
//    This library is free software; you can redistribute it and/or
//    modify it under the terms of the GNU Lesser General Public
//    License as published by the Free Software Foundation; either
//    version 2.1 of the License, or (at your option) any later version.
//
//    This library is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//    Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with this library; if not, write to the Free Software
//    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
//

namespace Accord.Audio.Visualizations
{
    using System;
    using Accord.Math;
    
    using Accord.Audio.Windows;
    using Accord.Math.Transforms;

    /// <summary>
    ///   Spectogram representation of an audio <see cref="Signal"/>.
    /// </summary>
    /// 
    [Serializable]
    public class Spectrogram 
    {

        string title = "Spectrogram";
        int channel;
        int windowSize;
        IWindow window;

        double[] frequencies;
        double[][] magnitudes;

        SpectrogramWindowCollection windowCollection;



        /// <summary>
        ///   Gets the collection of windows of this spectrogram.
        /// </summary>
        /// 
        public SpectrogramWindowCollection Windows
        {
            get
            {
                if (windowCollection == null)
                {
                    var windows = new SpectrogramWindow[magnitudes.Length];
                    for (int i = 0; i < windows.Length; i++)
                        windows[i] = new SpectrogramWindow(this, i);
                    windowCollection = new SpectrogramWindowCollection(windows);
                }
                return windowCollection;
            }
        }

        /// <summary>
        ///   Gets the magnitude of the signals at each window.
        /// </summary>
        /// 
        public double [][] Magnitudes
        {
            get { return magnitudes; }
        }

        /// <summary>
        ///   Gets the frequencies represented in this spectogram.
        /// </summary>
        /// 
        public double[] Frequencies
        {
            get { return frequencies; }
        }

        /// <summary>
        /// Gets or sets the title of this spectogram.
        /// </summary>
        /// 
        public string Title
        {
            get { return title;  }
            set { title = value; }
        }

        /// <summary>
        ///   Constructs an empty histogram
        /// </summary>
        /// 
        public Spectrogram()
        {
        }

        /// <summary>
        ///   Constructs an empty histogram
        /// </summary>
        /// 
        /// <param name="signal">The signal to be windowed in the spectrogram.</param>
        /// <param name="channel">The channel from which this spectrogram should be computed from. Default is 0.</param>
        /// <param name="numberOfWindows">The number of windows that this signal should be divided into. Default is 1.</param>
        /// 
        public Spectrogram(Signal signal, int numberOfWindows = 1, int channel = 0)
        {
            this.windowSize = signal.Length / numberOfWindows;
            this.window = new RectangularWindow(this.windowSize);
            this.channel = channel;

            Compute(signal);
        }

        /// <summary>
        ///   Constructs an empty histogram
        /// </summary>
        /// 
        /// <param name="signal">The signal to be windowed in the spectrogram.</param>
        /// <param name="channel">The channel from which this spectrogram should be computed from. Default is 0.</param>
        /// <param name="window">The windowing function to be used to create this spectrogram. Default is <see cref="RectangularWindow"/>.</param>
        /// 
        public Spectrogram(Signal signal, IWindow window, int channel = 0)
        {
            this.windowSize = window.Length;
            this.window = window;
            this.channel = channel;

            Compute(signal);
        }

        /// <summary>
        ///   Computes (populates) a Spectrogram mapping with values from a signal. 
        /// </summary>
        /// 
        /// <param name="signal">The signal to be windowed in the spectrogram.</param>
        /// 
        public void Compute(Signal signal)
        {
            this.frequencies = FourierTransform2.GetFrequencyVector(windowSize, signal.SampleRate);

            Signal[] signals = signal.Split(window: window, stepSize: windowSize);

            this.magnitudes = new double[signals.Length][];
            for (int i = 0; i < signals.Length; i++)
            {
                ComplexSignal c = signals[i].ToComplex();
                c.ForwardFourierTransform();
                this.magnitudes[i] = FourierTransform2.GetMagnitudeSpectrum(c.GetChannel(channel));
            }
        }

        /// <summary>
        ///   Gets the frequency count for a given frequency band.
        /// </summary>
        /// 
        /// <param name="windowIndex">The index of the window.</param>
        /// <param name="frequency">The frequency band.</param>
        /// 
        public double GetFrequencyCount(int windowIndex, double frequency)
        {
            double[] frequencies = Frequencies;
            double[] ms = Magnitudes[windowIndex];

            int idx = frequencies.Subtract(frequency).Abs().ArgMin();
            return ms[idx];
        }
    }
}
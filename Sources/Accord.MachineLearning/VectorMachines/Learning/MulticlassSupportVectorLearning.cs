// Accord Machine Learning Library
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

namespace Accord.MachineLearning.VectorMachines.Learning
{
    using System;
    using Accord.MachineLearning;
    using Accord.Math.Optimization.Losses;
    using Accord.Statistics.Kernels;
    using Accord.Statistics;
    
    using System.Collections.Concurrent;
    using System.Threading.Tasks;

    using InnerParameters = InnerParameters<SupportVectorMachine<Accord.Statistics.Kernels.IKernel<double[]>>, double[]>;
    using InnerLearning = ISupervisedLearning<SupportVectorMachine<Accord.Statistics.Kernels.IKernel<double[]>>, double[], bool>;

    /// <summary>
    ///   One-against-one Multi-class Support Vector Machine Learning Algorithm
    /// </summary>
    /// 
    /// <remarks>
    /// <para>
    ///   This class can be used to train Kernel Support Vector Machines with
    ///   any algorithm using a <c>one-against-one</c> strategy. The underlying 
    ///   training algorithm can be configured by defining the <see cref="OneVsOneLearning{TInput, TBinary, TInput}.Learner"/>
    ///   property.</para>
    ///   
    /// <para>
    ///   One example of learning algorithm that can be used with this class is the
    ///   <see cref="SequentialMinimalOptimization{TKernel}">Sequential Minimal Optimization
    ///   </see> (SMO) algorithm.</para>
    /// </remarks>
    /// 
    /// <example>
    /// <para>
    ///   The following example shows how to learn a linear, multi-class support vector 
    ///   machine using the <see cref="LinearDualCoordinateDescent"/> algorithm. </para>
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\VectorMachines\MulticlassSupportVectorLearningTest.cs" region="doc_learn_ldcd" />
    /// 
    /// <para>
    ///   The following example shows how to learn a non-linear, multi-class support 
    ///   vector machine using the <see cref="Gaussian"/> kernel and the 
    ///   <see cref="SequentialMinimalOptimization{TKernel}"/> algorithm. </para>
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\VectorMachines\MulticlassSupportVectorLearningTest.cs" region="doc_learn_gaussian" />
    ///   
    /// <para>
    ///   Support vector machines can have their weights calibrated in order to produce 
    ///   probability estimates (instead of simple class separation distances). The
    ///   following example shows how to use <see cref="ProbabilisticOutputCalibration"/>
    ///   within <see cref="MulticlassSupportVectorLearning{TKernel}"/> to generate a probabilistic
    ///   SVM:</para>
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\VectorMachines\MulticlassSupportVectorLearningTest.cs" region="doc_learn_calibration" />
    /// </example>
    /// 
    /// <seealso cref="SequentialMinimalOptimization{TKernel}"/>
    /// <seealso cref="MulticlassSupportVectorLearning{TKernel}"/>
    /// <seealso cref="MultilabelSupportVectorLearning{TKernel}"/>
    /// <seealso cref="SupportVectorMachine"/>
    /// 
    public class MulticlassSupportVectorLearning<TKernel, TInput> :
        BaseMulticlassSupportVectorLearning<TInput,
            SupportVectorMachine<TKernel, TInput>, TKernel,
            MulticlassSupportVectorMachine<TKernel, TInput>>
        where TKernel : IKernel<TInput>
    {
        /// <summary>
        /// Creates an instance of the model to be learned. Inheritors
        /// of this abstract class must define this method so new models
        /// can be created from the training data.
        /// </summary>
        protected override MulticlassSupportVectorMachine<TKernel, TInput> Create(int inputs, int classes)
        {
            return new MulticlassSupportVectorMachine<TKernel, TInput>(inputs, Kernel, classes);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MulticlassSupportVectorLearning{TKernel, TInput}"/> class.
        /// </summary>
        /// 
        public MulticlassSupportVectorLearning()
        {
            Learner = (p) => new SequentialMinimalOptimization<TKernel, TInput>();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MulticlassSupportVectorLearning{TKernel, TInput}"/> class.
        /// </summary>
        /// 
        public MulticlassSupportVectorLearning(MulticlassSupportVectorMachine<TKernel, TInput> machine)
        {
            Model = machine;

            Learner = (p) => new SequentialMinimalOptimization<TKernel, TInput>()
            {
                Model = Model[p.Pair.Class1, p.Pair.Class2]
            };
        }
    }

    // TODO: Probably those two will prove not to be necessary in the long term:

    /// <summary>
    ///   Base class for multi-class support vector learning algorithms.
    /// </summary>
    public abstract class BaseMulticlassSupportVectorLearning<TInput, TBinary, TKernel, TModel> :
        OneVsOneLearning<TInput, TBinary, TModel>
        where TBinary : SupportVectorMachine<TKernel, TInput>
        where TModel : OneVsOne<TBinary, TInput>
        where TKernel : IKernel<TInput>
    {
        /// <summary>
        ///   Gets or sets the kernel function to be used to learn the
        ///   <see cref="SupportVectorMachine{TKernel}">kernel support 
        ///   vector machines</see>.
        /// </summary>
        /// 
        //[Obsolete("Please the kernel function in the individual learner for each machine.")]
        public TKernel Kernel { get; set; }
    }

    /// <summary>
    ///   Base class for multi-class support vector learning algorithms.
    /// </summary>
    public abstract class BaseMulticlassSupportVectorLearning<TBinary, TKernel, TModel> :
        OneVsOneLearning<TBinary, TModel>
        where TBinary : SupportVectorMachine<TKernel>
        where TModel : OneVsOne<TBinary>
        where TKernel : IKernel<double[]>
    {
        /// <summary>
        ///   Gets or sets the kernel function to be used to learn the
        ///   <see cref="SupportVectorMachine{TKernel}">kernel support 
        ///   vector machines</see>.
        /// </summary>
        /// 
        //[Obsolete("Please the kernel function in the individual learner for each machine.")]
        public TKernel Kernel { get; set; }
    }
}

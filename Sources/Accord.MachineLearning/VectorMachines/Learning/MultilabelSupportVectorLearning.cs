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
    using Accord.Math;
    using System.Threading;
    using Accord.MachineLearning;
    using Accord.Statistics.Kernels;
    using Accord.Statistics;
    using Accord.Math.Optimization.Losses;
    
    using System.Threading.Tasks;

    using InnerParameters = InnerParameters<SupportVectorMachine<Accord.Statistics.Kernels.IKernel<double[]>>, double[]>;
    using InnerLearning = ISupervisedLearning<SupportVectorMachine<Accord.Statistics.Kernels.IKernel<double[]>>, double[], bool>;

    /// <summary>
    ///   One-against-all Multi-label Support Vector Machine Learning Algorithm
    /// </summary>
    /// 
    /// <remarks>
    /// <para>
    ///   This class can be used to train Kernel Support Vector Machines with
    ///   any algorithm using a <c>one-against-all</c> strategy. The underlying 
    ///   training algorithm can be configured by defining the <see cref="OneVsRestLearning{TInput, TBinary, TModel}.Learner"/>
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
    ///   The following example shows how to learn a linear, multi-label (one-vs-rest) support 
    ///   vector machine using the <see cref="LinearDualCoordinateDescent"/> algorithm. </para>
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\VectorMachines\MultilabelSupportVectorLearningTest.cs" region="doc_learn_ldcd" />
    /// 
    /// <para>
    ///   The following example shows how to learn a non-linear, multi-label (one-vs-rest) 
    ///   support vector machine using the <see cref="Gaussian"/> kernel and the 
    ///   <see cref="SequentialMinimalOptimization{TKernel}"/> algorithm. </para>
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\VectorMachines\MultilabelSupportVectorLearningTest.cs" region="doc_learn_gaussian" />
    ///   
    /// <para>
    ///   Support vector machines can have their weights calibrated in order to produce probability 
    ///   estimates (instead of simple class separation distances). The following example shows how 
    ///   to use <see cref="ProbabilisticOutputCalibration"/> within <see cref="MulticlassSupportVectorLearning{TKernel}"/> 
    ///   to generate a probabilistic SVM:</para>
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\VectorMachines\MultilabelSupportVectorLearningTest.cs" region="doc_learn_calibration" />
    /// </example>
    /// 
    /// <seealso cref="MultilabelSupportVectorLearning{TKernel}"/>
    /// <seealso cref="MulticlassSupportVectorMachine{TKernel}"/>
    /// 
    public class MultilabelSupportVectorLearning<TKernel> :
        BaseMultilabelSupportVectorLearning<double[],
            SupportVectorMachine<TKernel>, TKernel,
            MultilabelSupportVectorMachine<TKernel>>
        where TKernel : IKernel<double[]>
    {
        /// <summary>
        /// Creates an instance of the model to be learned. Inheritors
        /// of this abstract class must define this method so new models
        /// can be created from the training data.
        /// </summary>
        protected override MultilabelSupportVectorMachine<TKernel> Create(int inputs, int classes, bool multilabel)
        {
            return new MultilabelSupportVectorMachine<TKernel>(inputs, Kernel, classes)
            {
                Method = multilabel ? MultilabelProbabilityMethod.PerClass : MultilabelProbabilityMethod.SumsToOne
            };
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MultilabelSupportVectorLearning{TKernel}"/> class.
        /// </summary>
        public MultilabelSupportVectorLearning()
        {
            Learner = (p) => new SequentialMinimalOptimization<TKernel>();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MultilabelSupportVectorLearning{TKernel}"/> class.
        /// </summary>
        public MultilabelSupportVectorLearning(MultilabelSupportVectorMachine<TKernel> machine)
        {
            Model = machine;
            Learner = (p) => new SequentialMinimalOptimization<TKernel>()
            {
                Model = p.Model
            };
        }
    }

    /// <summary>
    ///   One-against-all Multi-label Support Vector Machine Learning Algorithm
    /// </summary>
    /// 
    /// <remarks>
    /// <para>
    ///   This class can be used to train Kernel Support Vector Machines with
    ///   any algorithm using a <c>one-against-all</c> strategy. The underlying 
    ///   training algorithm can be configured by defining the <see cref="OneVsRestLearning{TInput, TBinary, TModel}.Learner"/>
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
    ///   The following example shows how to learn a linear, multi-label (one-vs-rest) support 
    ///   vector machine using the <see cref="LinearDualCoordinateDescent"/> algorithm. </para>
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\VectorMachines\MultilabelSupportVectorLearningTest.cs" region="doc_learn_ldcd" />
    /// 
    /// <para>
    ///   The following example shows how to learn a non-linear, multi-label (one-vs-rest) 
    ///   support vector machine using the <see cref="Gaussian"/> kernel and the 
    ///   <see cref="SequentialMinimalOptimization{TKernel}"/> algorithm. </para>
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\VectorMachines\MultilabelSupportVectorLearningTest.cs" region="doc_learn_gaussian" />
    ///   
    /// <para>
    ///   Support vector machines can have their weights calibrated in order to produce probability 
    ///   estimates (instead of simple class separation distances). The following example shows how 
    ///   to use <see cref="ProbabilisticOutputCalibration"/> within <see cref="MulticlassSupportVectorLearning{TKernel}"/> 
    ///   to generate a probabilistic SVM:</para>
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\VectorMachines\MultilabelSupportVectorLearningTest.cs" region="doc_learn_calibration" />
    /// </example>
    /// 
    /// <seealso cref="MultilabelSupportVectorLearning{TKernel}"/>
    /// <seealso cref="MulticlassSupportVectorMachine{TKernel}"/>
    /// 
    public class MultilabelSupportVectorLearning<TKernel, TInput> :
        BaseMultilabelSupportVectorLearning<TInput,
            SupportVectorMachine<TKernel, TInput>, TKernel,
            MultilabelSupportVectorMachine<TKernel, TInput>>
        where TKernel : IKernel<TInput>
        where TInput : ICloneable
    {
        /// <summary>
        /// Creates an instance of the model to be learned. Inheritors
        /// of this abstract class must define this method so new models
        /// can be created from the training data.
        /// </summary>
        protected override MultilabelSupportVectorMachine<TKernel, TInput> Create(int inputs, int classes, bool multilabel)
        {
            return new MultilabelSupportVectorMachine<TKernel, TInput>(inputs, Kernel, classes)
            {
                Method = multilabel ? MultilabelProbabilityMethod.PerClass : MultilabelProbabilityMethod.SumsToOne
            };
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MultilabelSupportVectorLearning{TKernel, TInput}"/> class.
        /// </summary>
        public MultilabelSupportVectorLearning()
        {
            Learner = (p) => new SequentialMinimalOptimization<TKernel, TInput>();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MultilabelSupportVectorLearning{TKernel, TInput}"/> class.
        /// </summary>
        public MultilabelSupportVectorLearning(MultilabelSupportVectorMachine<TKernel, TInput> machine)
        {
            Model = machine;
            Learner = (p) => new SequentialMinimalOptimization<TKernel, TInput>()
            {
                Model = p.Model
            };
        }
    }

    // TODO: Probably this class will prove not to be necessary in the long term:

    /// <summary>
    ///   Base class for multi-label support vector learning algorithms.
    /// </summary>
    /// 
    public abstract class BaseMultilabelSupportVectorLearning<TInput, TBinary, TKernel, TModel> :
        OneVsRestLearning<TInput, TBinary, TModel>
        where TBinary : SupportVectorMachine<TKernel, TInput>
        where TModel : OneVsRest<TBinary, TInput>
        where TKernel : IKernel<TInput>
        where TInput : ICloneable
    {
        /// <summary>
        ///   Gets or sets the kernel function to be used to learn the
        ///   <see cref="SupportVectorMachine{TKernel}">kernel support 
        ///   vector machines</see>.
        /// </summary>
        /// 
        public TKernel Kernel { get; set; }
    }

}

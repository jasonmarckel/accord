﻿// Accord Core Library
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

namespace Accord.IO
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.IO;
    using System.IO.Compression;
    using System.Linq;
    using System.Reflection;
    using System.Runtime.CompilerServices;
    using System.Runtime.Serialization;
    using System.Runtime.Serialization.Formatters.Binary;
    using System.Text;
    using System.Text.Json;

    /// <summary>
    ///   Model serializer. Can be used to serialize and deserialize (i.e. save and 
    ///   load) models from the framework to and from the disk and other streams.
    /// </summary>
    /// 
    /// <remarks>
    ///   This class uses a binding mechanism to automatically convert files saved using 
    ///   older versions of the framework to the new format. If a deserialization doesn't 
    ///   work, please fill in a bug report at https://github.com/accord-net/framework/issues
    /// </remarks>
    /// 
    /// <example>
    /// <para>
    ///   The first example shows the simplest way to use the serializer to persist objects:</para>
    ///   <code source="Unit Tests\Accord.Tests.Core\SerializerTest.cs" region="doc_simple" />
    ///   
    /// <para>
    ///   The second example shows the same, but using compression:</para>
    ///   <code source="Unit Tests\Accord.Tests.Core\SerializerTest.cs" region="doc_compression" />
    ///   
    /// <para>
    ///   The third and last example shows a complete example on how to create, save and re-load
    ///   a classifier from disk using serialization:</para>
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\KNearestNeighbors\KNearestNeighborsTest.cs" region="doc_learn" />
    /// <code source="Unit Tests\Accord.Tests.MachineLearning\KNearestNeighbors\KNearestNeighborsTest.cs" region="doc_serialization" />
    /// </example>
    /// 
    public static class Serializer
    {
        const SerializerCompression DEFAULT_COMPRESSION = SerializerCompression.None;

        private static readonly Object lockObj = new Object();

        private static SerializerCompression ParseCompression(string path)
        {
            string ext = Path.GetExtension(path);
            if (ext == ".gz")
                return SerializerCompression.GZip;
            return SerializerCompression.None;
        }

        ///// <summary>
        /////   Saves an object to a stream.
        ///// </summary>
        ///// 
        ///// <param name="obj">The object to be serialized.</param>
        ///// <param name="stream">The stream to which the object is to be serialized.</param>
        ///// <param name="compression">The type of compression to use. Default is None.</param>
        ///// 
        //public static void Save<T>(this T obj, Stream stream, SerializerCompression compression = DEFAULT_COMPRESSION)
        //{
        //    Save(obj, new BinaryFormatter(), stream, compression);
        //}

        /// <summary>
        ///   Saves an object to a stream.
        /// </summary>
        /// 
        /// <param name="obj">The object to be serialized.</param>
        /// <param name="stream">The stream to which the object is to be serialized.</param>
        /// <param name="compression">The type of compression to use. Default is None.</param>
        /// 
        public static void Save<T>(this T obj, Stream stream, SerializerCompression compression = DEFAULT_COMPRESSION)
        {
            //if (formatter.SurrogateSelector == null)
            //    formatter.SurrogateSelector = GetSurrogate(typeof(T));

            if (compression == SerializerCompression.GZip)
            {
                using (var gzip = new GZipStream(stream, CompressionLevel.Optimal, leaveOpen: true))
                    JsonSerializer.Serialize(gzip, obj);
            }
            else if (compression == SerializerCompression.None)
            {
                JsonSerializer.Serialize(stream, obj);
            }
            else
            {
                throw new ArgumentException("compression");
            }
        }

        /// <summary>
        ///   Saves an object to a stream.
        /// </summary>
        /// 
        /// <param name="obj">The object to be serialized.</param>
        /// <param name="path">The path to the file to which the object is to be serialized.</param>
        /// 
        public static void Save<T>(this T obj, string path)
        {
            Save(obj, path, ParseCompression(path));
        }

        /// <summary>
        ///   Saves an object to a stream.
        /// </summary>
        /// 
        /// <param name="obj">The object to be serialized.</param>
        /// <param name="path">The path to the file to which the object is to be serialized.</param>
        /// <param name="compression">The type of compression to use. Default is None.</param>
        /// 
        public static void Save<T>(this T obj, string path, SerializerCompression compression = DEFAULT_COMPRESSION)
        {
            path = Path.GetFullPath(path);

            var dir = Path.GetDirectoryName(path);
            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            if (compression == SerializerCompression.GZip)
            {
                if (!path.EndsWith(".gz"))
                    path = path + ".gz";
            }

            using (var fs = new FileStream(path, FileMode.Create))
            {
                Save(obj, fs, compression);
            }
        }

        /// <summary>
        ///   Saves an object to a stream, represented as an array of bytes.
        /// </summary>
        /// 
        /// <param name="obj">The object to be serialized.</param>
        /// <param name="compression">The type of compression to use. Default is None.</param>
        /// 
        public static byte[] Save<T>(this T obj, SerializerCompression compression = DEFAULT_COMPRESSION)
        {
            byte[] bytes;
            Save(obj, out bytes, compression);
            return bytes;
        }

        /// <summary>
        ///   Saves an object to a stream.
        /// </summary>
        /// 
        /// <param name="obj">The object to be serialized.</param>
        /// <param name="bytes">The sequence of bytes to which the object has been serialized.</param>
        /// <param name="compression">The type of compression to use. Default is None.</param>
        /// 
        public static void Save<T>(this T obj, out byte[] bytes, SerializerCompression compression = DEFAULT_COMPRESSION)
        {
            using (var fs = new MemoryStream())
            {
                Save(obj, fs, compression);
                fs.Seek(0, SeekOrigin.Begin);
                bytes = fs.ToArray();
            }
        }

        ///// <summary>
        /////   Loads an object from a stream.
        ///// </summary>
        ///// 
        ///// <param name="stream">The stream from which the object is to be deserialized.</param>
        ///// <param name="compression">The type of compression to use. Default is None.</param>
        ///// 
        ///// <returns>The deserialized machine.</returns>
        ///// 
        //public static T Load<T>(Stream stream, SerializerCompression compression = DEFAULT_COMPRESSION)
        //{
        //    return Load<T>(stream, new BinaryFormatter(), compression);
        //}

        /// <summary>
        ///   Loads an object from a file.
        /// </summary>
        /// 
        /// <param name="path">The path to the file from which the object is to be deserialized.</param>
        /// 
        /// <returns>The deserialized object.</returns>
        /// 
        public static T Load<T>(string path)
        {
            return Load<T>(path, ParseCompression(path));
        }

        /// <summary>
        ///   Loads an object from a file.
        /// </summary>
        /// 
        /// <param name="path">The path to the file from which the object is to be deserialized.</param>
        /// <param name="compression">The type of compression to use. Default is None.</param>
        /// 
        /// <returns>The deserialized object.</returns>
        /// 
        public static T Load<T>(string path, SerializerCompression compression = DEFAULT_COMPRESSION)
        {
            path = Path.GetFullPath(path);
            return load<T>(path, compression);
        }

        private static T load<T>(string path, SerializerCompression compression)
        {
            using (var fs = new FileStream(path, FileMode.Open))
            {
                return Load<T>(fs, compression);
            }
        }

        /// <summary>
        ///   Loads an object from a stream, represented as an array of bytes.
        /// </summary>
        /// 
        /// <param name="bytes">The byte stream containing the object to be deserialized.</param>
        /// <param name="compression">The type of compression to use. Default is None.</param>
        /// 
        /// <returns>The deserialized object.</returns>
        /// 
        public static T Load<T>(byte[] bytes, SerializerCompression compression = DEFAULT_COMPRESSION)
        {
            using (var fs = new MemoryStream(bytes, false))
            {
                return Load<T>(fs, compression);
            }
        }



        ///// <summary>
        /////   Loads an object from a stream.
        ///// </summary>
        ///// 
        ///// <param name="stream">The stream from which the object is to be deserialized.</param>
        ///// <param name="compression">The type of compression to use. Default is None.</param>
        ///// <param name="value">The object to be read. This parameter can be used to avoid the
        /////   need of specifying a generic argument to this function.</param>
        ///// 
        ///// <returns>The deserialized machine.</returns>
        ///// 
        //public static T Load<T>(Stream stream, out T value, SerializerCompression compression = DEFAULT_COMPRESSION)
        //{
        //    return value = Load<T>(stream, compression);
        //}

        /// <summary>
        ///   Loads an object from a file.
        /// </summary>
        /// 
        /// <param name="path">The path to the file from which the object is to be deserialized.</param>
        /// <param name="value">The object to be read. This parameter can be used to avoid the
        ///   need of specifying a generic argument to this function.</param>
        /// 
        /// <returns>The deserialized object.</returns>
        /// 
        public static T Load<T>(string path, out T value)
        {
            return Load<T>(path, out value, ParseCompression(path));
        }

        /// <summary>
        ///   Loads an object from a file.
        /// </summary>
        /// 
        /// <param name="path">The path to the file from which the object is to be deserialized.</param>
        /// <param name="compression">The type of compression to use. Default is None.</param>
        /// <param name="value">The object to be read. This parameter can be used to avoid the
        ///   need of specifying a generic argument to this function.</param>
        /// 
        /// <returns>The deserialized object.</returns>
        /// 
        public static T Load<T>(string path, out T value, SerializerCompression compression)
        {
            return value = Load<T>(path, compression);
        }

        /// <summary>
        ///   Loads an object from a stream, represented as an array of bytes.
        /// </summary>
        /// 
        /// <param name="bytes">The byte stream containing the object to be deserialized.</param>
        /// <param name="value">The object to be read. This parameter can be used to avoid the
        ///   need of specifying a generic argument to this function.</param>
        /// <param name="compression">The type of compression to use. Default is None.</param>
        /// 
        /// <returns>The deserialized object.</returns>
        /// 
        public static T Load<T>(byte[] bytes, out T value, SerializerCompression compression = DEFAULT_COMPRESSION)
        {
            return value = Load<T>(bytes, compression);
        }


        /// <summary>
        ///   Loads a model from a stream.
        /// </summary>
        /// 
        /// <typeparam name="T">The type of the model to be loaded.</typeparam>
        /// <param name="stream">The stream from which to deserialize the object graph.</param>
        /// <param name="compression">The type of compression to use. Default is None.</param>
        /// 
        /// <returns>The deserialized object.</returns>
        /// 
        public static T Load<T>(Stream stream, SerializerCompression compression = DEFAULT_COMPRESSION)
        {
            lock (lockObj)
            {
                try
                {
                    AppDomain.CurrentDomain.AssemblyResolve += resolve;

                    object obj;
                    if (compression == SerializerCompression.GZip)
                    {
                        using (var gzip = new GZipStream(stream, CompressionMode.Decompress, leaveOpen: true))
                            obj = JsonSerializer.Deserialize<T>(gzip);
                    }
                    else if (compression == SerializerCompression.None)
                    {
                        obj = JsonSerializer.Deserialize<T>(stream);
                    }
                    else
                    {
                        throw new ArgumentException("compression");
                    }

                    if (obj is T)
                        return (T)obj;
                    return obj.To<T>();
                }
                finally
                {
                    AppDomain.CurrentDomain.AssemblyResolve -= resolve;
                }
            }
        }

        /// <summary>
        ///   Performs a deep copy of an object by serializing and deserializing it.
        /// </summary>
        /// 
        /// <typeparam name="T">The type of the model to be copied.</typeparam>
        /// <param name="obj">The object.</param>
        /// 
        /// <returns>A deep copy of the given object.</returns>
        /// 
        public static T DeepClone<T>(this T obj)
        {
            return Load<T>(Save<T>(obj));
        }

        private static Assembly resolve(object sender, ResolveEventArgs args)
        {
            var display = new AssemblyName(args.Name);

            if (display.Name == args.Name)
                return null;

            return ((AppDomain)sender).Load(display.Name);
        }


        /// <summary>
        ///   Retrieves a value from the SerializationInfo store.
        /// </summary>
        /// 
        /// <typeparam name="T">The type of the value to be retrieved.</typeparam>
        /// <param name="info">The serialization info store containing the value.</param>
        /// <param name="name">The name of the value.</param>
        /// <param name="value">The value retrieved from the info store.</param>
        /// 
        /// <returns>The value retrieved from the info store.</returns>
        /// 
        public static T GetValue<T>(this SerializationInfo info, string name, out T value)
        {
            return value = (T)info.GetValue(name, typeof(T));
        }
    }
}

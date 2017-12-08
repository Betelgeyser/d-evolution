/**
 * Copyright © 2017 Sergei Iurevich Filippov, All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
module network;

import layer;

struct RandomNetwork
{
	InputLayer  inputLayer;
	HiddenLayer outputLayer;
	private RandomLayer[] hiddenLayers;
	
	this(T)(ulong inputs, ulong outputs, ulong maxHLayers, ulong maxNeurons, double minWeight, double maxWeigth, T generator)
	{
		assert (maxHLayers >= 1);
		
		inputLayer = InputLayer(inputs);
		hiddenLayers ~= RandomLayer(maxNeurons, inputLayer.length, minWeight, maxWeigth, generator);
		
		for(ulong i = 1; i < maxHLayers; i++)
			hiddenLayers ~= RandomLayer(maxNeurons, hiddenLayers[i - 1].length, minWeight, maxWeigth, generator);
		
		outputLayer = HiddenLayer(outputs, hiddenLayers[hiddenLayers.length - 1].length, minWeight, maxWeigth, generator);
		outputLayer.sig = false;
	}
}
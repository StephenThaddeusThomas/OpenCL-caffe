/**********************************************************************
Copyright ?2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

?	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
?	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
// to share code with between CPU and GPU

#include "aDNNInternal.hpp"


namespace adnn
{


	/************************************************************************************************************************
	**
	**			ADNN  network
	**
	************************************************************************************************************************/



	/**
	* Constructors
	*/
	ADNN::ADNN(const aDNNode & net, const adnn_net_parameters & node_params)
		:aDNNode(net, node_params)
	{
	}
	// root - net - creation
	// root - net - creation

	ADNN::ADNN(const ADNNLib & lib, const  adnn_net_parameters & node_params)
	{
		// initialize node
		setParent((void*)&lib);
		init(node_params);
	}


	ADNN::ADNN()
		: aDNNode()
	{

	}


	ADNN::ADNN(const ADNN & rh)
	{
		*this = rh;
	}

	const aDNNode & ADNN:: operator = (const ADNN & rh)
	{
		*(aDNNode*)this = *(aDNNode*)&rh;
		return *this;
	}

	/**
	* Destructor
	*/

	ADNN::~ADNN(void)
	{


		for (std::vector<aDNNode *>::iterator i = net_owned_.begin(); i != net_owned_.end(); ++i)
		{
			(*i)->release();
		}

	}


	aDNNode * ADNN::AddNode(const adnn_node_parameters & node_descr)
	{
		aDNNode * newNode = NULL;
		ADNNBase * base = (ADNNBase *)getParent();

		// use prevous version for now
		if (getType() == ADNN_NODE_NET)
		{

			switch (node_descr.type)
			{

			case ADNN_NODE_CONV:
				newNode = new aDNNodeConv(*base, node_descr);;
				break;
			case ADNN_NODE_NEURON:
				newNode = new aDNNodeNeuron(*base, node_descr);
				break;
			case ADNN_NODE_POOLING:
				newNode = new aDNNodePooling(*base, node_descr);
				break;
			case ADNN_NODE_RESP_NORM:
				newNode = new aDNNodeLRN(*base, node_descr);
				break;
			case ADNN_NODE_FULLY_CONNECT:
				newNode = new aDNNodeFullyConnect(*base, node_descr);
				break;
			case ADNN_NODE_SOFTMAX:
				newNode = new aDNNodeSoftMax(*base, node_descr);
				break;
			case ADNN_NODE_SOFTMAX_COST_CROSSENTROPY:
				newNode = new aDNNodeSoftMax(*base, node_descr);
				assert(newNode);
				((aDNNodeSoftMax*)newNode)->setCrossEntrypyLoss(true);
				break;

			default:
				printf("ERROR unknown node : %d\n", node_descr.type);
				break;

			}

		}
		if (newNode)
		{

			// nodes has to be deleted explicitely
			net_owned_.push_back(newNode);
			net_.push_back(newNode);

			newNode->setNet(this);
			printf("Created Node: %s\n", newNode->getName().c_str());
		}
		else
		{
			printf("Unknown error in AddNode: %s\n", newNode->getName().c_str());

		}

		return(newNode);
	}


	int ADNN::AddNodes(int n_nodes, aDNNode ** nodes)
	{
		int ret = 0;
		for (int i = 0; i < n_nodes; ++i)
		{

			// nodes has to be deleted explicitely
			net_.push_back(nodes[i]);
			nodes[i]->setNet(this);


		}

		return(ret);
	}


	int ADNN::Connect(void)
	{
		int ret = 0;
		bool found_src = false;
		// get all input connection and find source and sink
		for (std::vector<aDNNode *>::iterator i = net_.begin(); i != net_.end(); ++i)
		{

// over all my inputs
			for (std::vector<aDNNEdge>::iterator k = (**i).inputs_.begin(); k != (**i).inputs_.end(); ++k)
			{
// find me as input
				std::string my_input = (*k).getName();
				bool found_input = false;
				for (std::vector<aDNNode *>::iterator l = net_.begin(); l != net_.end(); ++l)
				{
					std::string node_nm = (**l).getName();
					if (my_input.compare(node_nm) == 0)
					{
						found_input = true;
// I'm its input node
						(**i).setInputNode(l, (int)(k - (**i).inputs_.begin()));
						break;
					}

				}
				if (!found_input )
				{

// I'm external input and designate me as such
					if ((*k).getEdgeType() == ADNN_ED_SOURCE)
					{
						found_src = true;
					}
				}
			}


			if (!found_src)
			{
				printf("Net construction error: cannot find source\n");
				ret = -1;
				return(ret);
			}

			// find my output

			std::string my_name = (**i).getName();

			bool found_output = false;
			for (std::vector<aDNNode *>::iterator k = net_.begin(); k != net_.end(); ++k)
			{
				std::vector<aDNNEdge>::iterator inp = (**k).getInputByName(my_name);
				if (inp != (**k).getInputEdges().end())
				{
					(**i).setOutputNode(k);
					found_output = true;
					break;
				}

			}

			// i'm not input to anybody
			// i'm sink
			if (!found_output)
			{
				// I'm sink input and designate me as such
				(**i).setOutputEdgeType(ADNN_ED_SINK);
				// sink node
				(**i).setOutputEdgeName(my_name);

			}
		}

// verify there is a single source and single sink
		int src_count = 0;
		int sink_count = 0;
		for (std::vector<aDNNode *>::iterator i = net_.begin(); i != net_.end(); ++i)
		{
			src_count += (int)((**i).getNInternalInputEdges() == 0);
			sink_count += (int)((**i).getNInternalOutputEdges() == 0);
		}

		if (src_count > 1 || sink_count > 1)
		{
			printf("Net construction error: more than 1 source or sink\n");
			ret = -1;
			return(ret);
		}
// find source and sink and make the input output edges of the net
		else
		{
			for (std::vector<aDNNode *>::iterator i = net_.begin(); i != net_.end(); ++i)
			{
				if ((**i).getNInternalInputEdges() == 0)
				{
					setInputNode(i);
				}
				if ((**i).getNInternalOutputEdges() == 0)
				{
					setOutputNode(i);
				}
			}

		}

		return(ret);
	}


	int ADNN::Construct(void)
	{
		int ret = 0;
// start from the bottom
		std::vector<aDNNode *>::iterator i = getInputNode();
		while (true)
		{

			(**i).Construct();
			if (i == getOutputNode())
			{
				break;
			}
			i = (**i).getOutputNode();

		};
		return(ret);
	}

	int ADNN::ConstructBwd(void)
	{
// run in reverse order
		int ret = 0;
		// start from the top
		std::vector<aDNNode *>::iterator i = getOutputNode();
		while (true)
		{

			(**i).ConstructBwd();
			if (i == getInputNode())
			{
				break;
			}
			i = (**i).getInputNode();

		};
		return(ret);
	}



	int ADNN::Build(void)
	{
		int ret = 0;
		// start from the bottom
		std::vector<aDNNode *>::iterator i = getInputNode();
		while (true)
		{

			(**i).Build();
			if (i == getOutputNode())
			{
				break;
			}
			i = (**i).getOutputNode();

		};
		return(ret);
	}


	int ADNN::BuildBwd(void)
	{
		int ret = 0;
		// start from the top
		std::vector<aDNNode *>::iterator i = getOutputNode();
		while (true)
		{

			(**i).BuildBwd();
			if (i == getInputNode())
			{
				break;
			}
			i = (**i).getInputNode();

		};
		return(ret);
	}


	int ADNN::Run(bool forward, int n_running_params, const adnn_node_parameters * running_params)
	{
		int ret = 0;
		// start from the bottom
		if (forward)
		{
			ret = RunFwd(n_running_params, running_params);
		}

		return(ret);
	}


	int ADNN::RunFwd(int n_running_params, const adnn_node_parameters * running_params)
	{
		int ret = 0;
		// start from the bottom
		std::vector<aDNNode *>::iterator i = getInputNode();
		while (true)
		{
			const adnn_node_parameters * my_running_params = NULL;
			for (int j = 0; j < n_running_params && running_params; ++j)
			{
				if ((**i).getName().compare(std::string(running_params[j].name)) == 0)
				{
					my_running_params = &running_params[j];
				}
			}

			ret = (**i).RunFwd(my_running_params);
			if (i == getOutputNode() || ret != 0)
			{
				break;
			}
			i = (**i).getOutputNode();

		};
		return(ret);
	}

	int ADNN::RunBwd(int n_running_params, const adnn_node_parameters * running_params)
	{
		int ret = 0;
		// start from the top
		std::vector<aDNNode *>::iterator i = getOutputNode();
		while (true)
		{
			const adnn_node_parameters * my_running_params = NULL;
			for (int j = 0; j < n_running_params && running_params; ++j)
			{
				if ((**i).getName().compare(std::string(running_params[j].name)) == 0)
				{
					my_running_params = &running_params[j];
				}
			}

			ret = (**i).RunBwd(my_running_params);
			if (i == getInputNode() || ret != 0)
			{
				break;
			}
			i = (**i).getInputNode();

		};
		return(ret);
	}


	/************************************************************************************************************************
	**
	**				UPDATE WEIGHTS
	**
	************************************************************************************************************************/

	// can be done on the different queue

	int ADNN::UpdateWeights(void)
	{
		int ret = 0;
		// start from the bottom
		std::vector<aDNNode *>::iterator i = getInputNode();
		while (true)
		{

			ret = (**i).UpdateWeights();
			if (i == getOutputNode() || ret != 0)
			{
				break;
			}
			i = (**i).getOutputNode();

		};

		// update counter
		setInternalCounter(getInternalCounter() + 1);
		return(ret);
	}

	int ADNN::RemoveNode(aDNNode * node)
	{
		int ret = 0;
// find in the net_
		std::vector<aDNNode*> ::iterator net_i = ADNN::findNode(node);
// find in the owned
		std::vector<aDNNode*> ::iterator net_owned_i = net_owned_.end();
		for (std::vector<aDNNode *>::iterator i = net_owned_.begin(); i != net_owned_.end(); ++i)
		{
			if (*i == node)
			{
				net_owned_i = i;
				break;
			}
		}

// delete if owned
		if (net_owned_i != net_owned_.end())
		{
			node->release();
		}


		net_.erase(net_i);
		net_owned_.erase(net_owned_i);

		return(ret);
	}

	std::vector<aDNNode*> ::iterator ADNN::findNode(aDNNode * node)
	{
		std::vector<aDNNode*> ::iterator ret = net_.end();
		for (std::vector<aDNNode *>::iterator i = net_.begin(); i != net_.end(); ++i)
		{
			if (*i == node)
			{
				ret = i;
				break;
			}
		}
		return (ret);
	}




} // adnn







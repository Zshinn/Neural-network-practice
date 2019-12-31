
#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/core/CHeader.h"
using namespace nts;

namespace myNetwork
{
	struct mymodel
	{
		XTensor w1;

		XTensor w2;

		XTensor y;

		int h_size;

		int devID;
	};

	struct mynet
	{
		XTensor input;
		// w1 * input
		XTensor hidden_state1;
		// sigmoid(hidden_state1)
		XTensor hidden_state2;
		// w2 * hidden_state2
		XTensor hidden_state3;
		/*output*/
		XTensor output;
	};
	int myNetMain(int argc, const char ** argv);
};


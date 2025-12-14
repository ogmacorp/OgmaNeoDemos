#include <aogmaneo/hierarchy.h>
#include "aon_utils.hpp"

#include <cmath>

#include <time.h>
#include <iostream>
#include <fstream>

#include "csdrScalarEncoder.hpp"
#include "cartpole/cartpole.hpp"
using namespace std;
using namespace aon;

/*
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.

	For the interested reader:
    https://coneural.org/florian/papers/05_cart_pole.pdf
*/


#include "getopt.h"
int main(int argc, char *argv[])
{
	bool loadHierarchy = false;
	bool learnFlag = true;
  std::string hFileName   = "CartPole.ohr";

	int opt;
	while ((opt = getopt(argc, argv, "l:h")) != -1) {  // for each option...
		switch (opt) {
		case 'l':
			learnFlag = std::stoi(optarg);
			break;
		case 'h':
			loadHierarchy = true;
			break;
		case '?':
			std::cerr << "valid option -l 1 or -h 0 !" << std::endl;
			break;
		}
	}

	// Create hierarchy
	set_num_threads(4);
	const int numLayers				= 2;
	const int numInputs				= 2;
	const int numActions			= 2;
	const int numSensors			= 4;
	const int rootNumSensors	= (int)std::sqrt(numSensors); // std::ceil()

    // for encoding/decoding scalar input
    const int sensorResolution= 16; //Resolution (column size) of encoding

    // --------------------------- Create the Hierarchy ---------------------------
    const int eRadius                   = 2;    // encoder radius
    const int dRadius                   = 2;    // decoder radius
    const int num_dendrites_per_cell    = 4;
    const int history_capacity          = 256;

	const int ticks_per_update = 2; // number of ticks a layer takes to update (relative to previous layer)
    const int temporal_horizon = 2; // temporal distance into the past addressed by the layer. should be greater than or equal to ticks_per_update

	Hierarchy h;
	Array<Hierarchy::Layer_Desc> lds(numLayers);
	for (int i = 0; i < lds.size(); i++) {
		lds[i].hidden_size 				= Int3(4, 4, 16);
		lds[i].num_dendrites_per_cell   = num_dendrites_per_cell;
        lds[i].ticks_per_update         = ticks_per_update;
        lds[i].temporal_horizon         = temporal_horizon;
	}

	// here we use the current 1.sensor data and no prediction  --> InputType = none
	//             the 2.sensor is action, and it is actor 			--> InputType = action
	Array<Hierarchy::IO_Desc> ioDescs(numInputs);
	ioDescs[0] = Hierarchy::IO_Desc(Int3(rootNumSensors, rootNumSensors, sensorResolution), IO_Type::prediction, num_dendrites_per_cell, 8, eRadius, dRadius, history_capacity);
	ioDescs[1] = Hierarchy::IO_Desc(Int3(1, 1, numActions), IO_Type::action, num_dendrites_per_cell, 8, eRadius, dRadius, history_capacity);

	if (loadHierarchy)
	{
		std::cout << "load hierarchy" << std::endl;
		CustomerStreamReader reader;
		reader.ins.open(hFileName.c_str(), std::ios::binary);
		h.read(reader);
		learnFlag = false;
	}
	else
	{
		h.init_random(ioDescs, lds);
		// Set some parameters for the actor IO layer (index 1)
		//h.getALayer(1).vlr = 0.01;
		//h.getALayer(1).alr = 0.01;
		//h.getALayer(1).discount = 0.99;
		//h.getALayer(1).minSteps = 16;
		//h.getALayer(1).historyIters = 16;
		//h.setImportance(1, 0.0f); // Don't need to pay attention to the action for this task
	}

	int numColumns = 9;	// == sensorResolution????
#ifdef USE_SCALAR_ENCODER_
	// worse performance than binningEncoder()
	int numCellsPerColumn = 16;
	cpScalarEncoder enc(numSensors, numColumns, numCellsPerColumn, 0, +1);
#endif

	int n_episodes = 1000; // Number of episodes
	CartPoleEnv env;

	float reward = 0.0;
	int8_t action = 0;
	bool done = false;

	int t = 0, bestTimeSteps = -1, bestEpisode;
	// training
	if (learnFlag)
	{
		for (auto episode = 0; episode < n_episodes; ++episode)
		{
			// obs = [x, x_dot, theta, theta_dot]
			auto obs = env.reset(); // Reset the environment (as OpenAI's gym)
			t = 0;
			done = false;

//#define _USE_GLOBAL_SENSOR_ACTION_INPUT_
#ifdef _USE_GLOBAL_SENSOR_ACTION_INPUT_
			// maybe it is because the hierarchy of OgmaNeo, e.g. h.step(using_address_of_inputs)
			// BUT it does NOT work correctly --> WHY????????????????
			// and worse than (not use _USE_GLOBAL_SENSOR_ACTION_INPUT_  and not use USE_DEFAULT_PREDICTION)
			Array<Int_Buffer_View> inputCIs(ioDescs.size());
			inputCIs[1] = h.get_prediction_cis(1); // If not modifying the action beyond the default exploration, we can just pass in the prediction directly
#endif

			while (!done)
			{
				++t;
				//std::cout << "obss: " << obs[0] << ", " << obs[1] << ", " << obs[2] << ", " << obs[3] << "\n";
				// encode obss into CSDR
#ifdef USE_SCALAR_ENCODER_
				std::vector<int> csdr = enc.encode({sigmoid(4.f*obs[0]), sigmoid(4.f*obs[1]), sigmoid(4.f*obs[2]), sigmoid(4.f*obs[3])});
#else
				std::vector<int> csdr = binningEncoder(obs, sensorResolution, 4.f);
#endif

#ifdef _USE_GLOBAL_SENSOR_ACTION_INPUT_
				Int_Buffer sensorCIs(csdr.size(), 0);
				for (auto i = 0; i < csdr.size(); ++i) sensorCIs[i] = csdr[i];
				inputCIs[0] = sensorCIs;
				//inputCIs[1] = &h.get_prediction_cis(1); 	// no influence!!!
#else
				Array<Int_Buffer_View> inputCIs(ioDescs.size());
				Int_Buffer sensorCIs(csdr.size(), 0);
				for (auto i = 0; i < csdr.size(); ++i) sensorCIs[i] = csdr[i];
				inputCIs[0] = sensorCIs;
				auto testB = h.get_prediction_cis(1);
				if (testB.size() != 1 || testB[0] != action) std::cout << "--> false action data!!!!\n";
//#define USE_DEFAULT_PREDICTION
#ifdef USE_DEFAULT_PREDICTION
				// it does NOT work correctly --> WHY????????????????
				inputCIs[1] = h.get_prediction_cis(1);
				if (inputCIs[1]->size() != 1 || (*inputCIs[1])[0] != action) std::cout << "--> AFTER false action data!!!!\n";
#else
				Int_Buffer actionCIs(1, action);
				inputCIs[1] = actionCIs;
#endif  // USE_DEFAULT_PREDICTION

#endif	//_USE_GLOBAL_SENSOR_ACTION_INPUT_

				// run hierarchy by current sensor input, current action/rewards
				h.step(inputCIs, learnFlag, reward);

				// Retrieve the action, the hierarchy already automatically applied exploration
				action = h.get_prediction_cis(1)[0]; // First and only column

				std::tie(obs, reward, done) = env.step(action); // Step the environment (as OpenAI's gym)

				//std::cout << "action: " << int(action) << "-> obs: " << obs[0] << " ," << obs[1] << " ," << obs[2] << " ," << obs[3] << std::endl;
				if (done)
				{
					reward = -100.0;

					if (t > bestTimeSteps)
					{
						bestTimeSteps = t;
						bestEpisode   = episode;

						// save the best states
						CustomerStreamWriter writer;
						writer.outs.open(hFileName.c_str(), std::ios::out | std::ios::binary);
						h.write(writer);
					}
					std::cout << "Episode: " << episode << " finished after " << t << " time steps / best: " << bestTimeSteps << " at episode: " << bestEpisode << std::endl;
				}
				else
					reward = 0;
			}
		}
	}

	std::cout << "best training time steps: " << bestTimeSteps << " at episode: " << bestEpisode << std::endl;

	// testing
	std::cout << "..now testing ...\n";
	learnFlag = false;
	action = 0;
	reward = 0;
	t = 0;
	done = false;
	bool extraEpisode = false; 	// in the result testing to average out randomness.
	auto obs = env.reset(); 		// Reset the environment (as OpenAI's gym)
	while (!done || extraEpisode)
	{
		if (extraEpisode) std::cout << "run extra episode to average out randomness" << std::endl;
		t++;
		//std::cout << "x: " << obs[0] << ", theta: " << obs[2] << "\n";
		// encode obss into CSDR
#ifdef USE_SCALAR_ENCODER_
				std::vector<int> csdr = enc.encode({sigmoid(4.f*obs[0]), sigmoid(4.f*obs[1]), sigmoid(4.f*obs[2]), sigmoid(4.f*obs[3])});
#else
				std::vector<int> csdr = binningEncoder(obs, sensorResolution, 4.f);
#endif
		Array<Int_Buffer_View> inputCIs(ioDescs.size());
		Int_Buffer sensorCIs(csdr.size(), 0);
		for (auto i = 0; i < csdr.size(); ++i) sensorCIs[i] = csdr[i];
		inputCIs[0] = sensorCIs;
		Int_Buffer actionCI(1, action);
		inputCIs[1] = actionCI;

		// run hierarchy by current sensor input, current action/rewards
		h.step(inputCIs, learnFlag, reward);

		// Retrieve the action, the hierarchy already automatically applied exploration
		action = h.get_prediction_cis(1)[0]; // First and only column

		std::tie(obs, reward, done) = env.step(action); // Step the environment (as OpenAI's gym)

		if (done)
		{
			std::cout << "x: " << obs[0] << " in [-4, +4]?, theta: " << obs[2]*180/M_PIf32 << " degrees\n";
			std::cout << "Episode finished after " << t << " time steps" << std::endl;
			extraEpisode = !extraEpisode;	//run a extra episode
			reward = -100;
			t=0;
		}
	}

	return 1;
}


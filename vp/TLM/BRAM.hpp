#ifndef BRAM_H
#define BRAM_H
#define SC_INCLUDE_FX

#include <iostream>
#include <systemc>
#include <string>
#include <fstream>
#include <deque>
#include <vector>
#include <array>
#include <algorithm>
#include "types.hpp"
#include "addresses.hpp"
#include "tlm_utils/tlm_quantumkeeper.h"

using namespace std;
using namespace sc_core;

SC_MODULE(BRAM)
{
public:
	SC_HAS_PROCESS(BRAM);
	BRAM(sc_module_name);

	tlm_utils::simple_target_socket<BRAM> s_bram_t0;
	tlm_utils::simple_target_socket<BRAM> s_bram_t1;

protected:
	void b_transport0(pl_t&,sc_time&);
	void proc();
	
	vector <hwdata_t> BRAM_cell;
	 

};

#endif
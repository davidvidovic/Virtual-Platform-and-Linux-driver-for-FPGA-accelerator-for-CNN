#ifndef BRAM_C
#define BRAM_C
#include "BRAM.hpp"

BRAM::BRAM(sc_module_name):sc_module(name)
{
	SC_THREAD(proc);
	s_bram_t0.register_b_transport(this, &Hardware::b_transport);
	s_bram_t1.register_b_transport(this, &Hardware::b_transport);
	BRAM_cell.reserve(1024); //in each bram cell max 1024 16-bit elements 
}

void BRAM::proc()
{
	//do noting
}

void BRAM::b_transport(pl_t&,sc_time&)
{
	tlm_command cmd    = pl.get_command();
	uint64 adr         = pl.get_address();
	const unsigned char *buf = pl.get_data_ptr();
	unsigned int len   = pl.get_data_length();
	switch(cmd)
	{
		case TLM_WRITE_COMMAND:
			for(unsigned int i=0; i<len; i++)
                        {
                                BRAM_cell[adr+i]=((hwdata_t*)buf)[i];
                        }
                        pl.set_response_status(TLM_OK_RESPONSE);
			break;
		case TLM_READ_COMMAND:
			buf = (unsigned char*)&BRAM_cell[adr];
                        pl.set_data_ptr(buf);
                        pl.set_response_status(TLM_OK_RESPONSE);
			break;
		default:
			pl.set_response_status( TLM_COMMAND_ERROR_RESPONSE );
	}
	// offset += sc_time(10, SC_NS);
}



#endif
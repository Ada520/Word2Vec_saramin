#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <vector>

using namespace std;

int total_group_cnt;

class edge
{
public:
	edge (int from, int to)
	{
		this->from = from;
		this->to = to;
	}

	bool operator< (const edge& e) const
	{
		if (from == e.from)
			return to < e.to;
		else
			return from < e.from;
	}

	bool operator> (const edge& e) const
	{
		if (from == e.from)
			return to > e.to;
		else
			return from > e.from;
	}

	bool operator== (const edge& e) const
	{
		if (from == e.from && to == e.to)
			return true;
		else
			return false;
	}

	int from;
	int to;


};

int get_id (map<string, int>* id_map, vector<string>* id_2_string, string name)
{
	if (id_map->find (name) != id_map->end())
		return (*id_map)[name];
	else
	{
		int size = id_map->size();
		//cout<<"size : " << size << endl;
		(*id_map)[name] = size;
		id_2_string->push_back (name);

		return size;
	}
}

int assign_group (map<string, int>* group_map, string from, string name)
{
	if (group_map->find (name) == group_map->end())
	{
		if ((*group_map)[from] == 0)
			(*group_map)[name] = total_group_cnt++;
		else
			(*group_map)[name] = (*group_map)[from];
	}
	else
		return (*group_map)[name];
}

string get_seed (string file_name)
{
	ifstream ifs (file_name);
	int idx;
	string buffer;
	
	string from_str;

	// cast 1st line away
	ifs >> buffer;

	idx = buffer.find (',');
	from_str = buffer.substr (0, idx);

	ifs.close();

	return from_str;
}

void process_file (string file_name, map<string, int>* id_map, vector<string>* id_2_string, 
		   map<edge*, float>* edge_map, map<string, int>* group_map)
{
	ifstream ifs (file_name);
	int idx;
	string buffer;
	
	string from_str;
	string to_str;

	int from_id;
	int to_id;

	float sim;

	// cast 1st line away
	ifs >> buffer;

	while (!ifs.eof())
	{
		ifs >> buffer;

		//cout << "line read : " << buffer << endl;

		if (buffer.length() == 0)
			return;

		idx = buffer.find (',');
		from_str = buffer.substr (0, idx);
		//cout << "from_str : " << from_str << endl;
		buffer = buffer.substr (idx+1);
		//cout << "buffer remained : " << buffer << endl;

		idx = buffer.find (',');
		to_str = buffer.substr (0, idx);
		//cout << "to_str : " << to_str << endl;
		buffer = buffer.substr (idx+1);
		//cout << "buffer remained : " << buffer << endl;

		if (from_str == to_str)
			return;

		if (from_str == "source" && to_str == "target")
			continue;

		from_id = get_id (id_map, id_2_string, from_str);
		to_id = get_id (id_map, id_2_string, to_str);
		sim = stof(buffer);

		//cout << "from : " << from_id << ", to : " << to_id << ", sim : " << sim << endl;

		edge* e = new edge (from_id, to_id);
		(*edge_map)[e] = 1/sim;

		assign_group (group_map, from_str, to_str);
		// TODO ¤remove duplicated edge
	}
}



int main (int argc, char** argv)
{
	if (argc == 1)
	{
		//cout<<"args : program, input file names1, 2, 3 ..."<<endl;
		return 1;
	}

	map<string, int>* id_map = new map<string, int> ();
	map<edge*, float>* edge_map = new map<edge*, float> ();
	vector<string>* id_2_string = new vector<string> ();
	map<string, int>* group_map = new map<string, int> ();
	total_group_cnt = 0;

	// read first file, find seed
	string seed_str = get_seed (argv[1]);
	(*group_map)[seed_str] = total_group_cnt++;

	for (int i=1 ; i<argc ; i++)
		process_file (argv[i], id_map, id_2_string, edge_map, group_map);

	//cout << "file read done" << endl;

	ofstream ofs ("graph.json");

	ofs << "{" << endl;
	ofs << "\t\"nodes\":[" << endl;

	for (int i=0 ; i<id_map->size() ; i++)
	{
		ofs << "\t\t{" << endl;
		ofs << "\t\t\t\"name\":\"" << (*id_2_string)[i] << "\"," << endl;
		ofs << "\t\t\t\"group\":" << (*group_map)[(*id_2_string)[i]] << endl;

		//cout<<"current i : " << i << endl;

		if (i == id_map->size() - 1)
			ofs << "\t\t}" << endl;
		else
			ofs << "\t\t}," << endl;
	}
	ofs << "\t]," << endl;

	map<edge*, float>::iterator iter_edge = edge_map->begin();
	map<edge*, float>::iterator iter_edge_end = edge_map->end();

	ofs << "\t\"links\":[" << endl;

	bool not_first = false;
	int from_group, to_group;

	while (iter_edge != iter_edge_end)
	{
		from_group = (*group_map)[(*id_2_string)[iter_edge->first->from]];
		to_group = (*group_map)[(*id_2_string)[iter_edge->first->to]];

		if (from_group == to_group || from_group == 0)
		{
			if (not_first)
				ofs << ",\t\t{" << endl;
			else
				ofs << "\t\t{" << endl;

			ofs << "\t\t\t\"source\":" << iter_edge->first->from << "," << endl;
			ofs << "\t\t\t\"target\":" << iter_edge->first->to << "," << endl;
			ofs << "\t\t\t\"value\":" << iter_edge->second << endl;
			not_first = true;
			
			ofs << "\t\t}" << endl;
		}

		iter_edge++;
	}
	ofs << "\t]" << endl;
	ofs << "}" << endl;

	ofs.close();

	return 1;
}

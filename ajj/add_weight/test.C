void test(){
	ifstream infile("file18");
	string buffer;double xs;TString name;
//	infile>>name>>xs;
	while (!infile.eof()){
		infile>>name>>xs;
		if(name.Contains("end")==1)  break;
		cout<<name<<", "<<xs<<", "<<xs+1<<endl;
	}
}

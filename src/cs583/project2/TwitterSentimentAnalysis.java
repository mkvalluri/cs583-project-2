package cs583.project2;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

public class TwitterSentimentAnalysis {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Map<String, Integer> obamaTweets = new HashMap<String, Integer>();
		Map<String, Integer> romneyTweets = new HashMap<String, Integer>();
		System.out.println("Hello! Twitter");
		Path sourcePath = FileSystems.getDefault().getPath("data", "obama_new.csv");
		System.out.println(sourcePath.toAbsolutePath());
		int count = 0;
		// read file into stream, try-with-resources
		try {
			List<String> stream = Files.readAllLines(sourcePath);
			System.out.println(stream.size());
			for(String s : stream) {
				String[] temp = s.split(",");
				String t = "";
				for(int i = 0; i < temp.length - 2; i++) {
					t += temp[i];
				}
				count++;
				obamaTweets.put(t,Integer.parseInt(temp[temp.length - 1]));
			}
			
			System.out.println(obamaTweets.values().size() + ", " + count);

		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}

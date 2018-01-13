# okay simple julia script for the image scraping

include("image_scraper.jl")

#make results functino
base_results = "scrape_images"
if !isdir(base_results)
	mkdir(base_results)
end

function gestalt_scrape(basepath, num)
	keywords = ("gestalt", "gestalt continuation", "visual illusion", "gestalt image")
	for keyword in keywords
		if !isdir(keyword)
			mkdir(keyword)
		end
		scrape_images(keyword, num, basepath * "/" * keyword)
	end
end

gestalt_scrape(base_results, 500)





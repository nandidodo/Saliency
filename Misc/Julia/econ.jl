#just some quick thoughts on stuff perhaps?
#okay, so I was just having some quick thoughts about this... not sure whether
# I should try to turn it into a model or not, as this is theoretically 
#procrastination - and serious procrastination - from phd work... dagnabbit
# but I was thinking about the dutch tulip craze and bubble and how bubbles generally work
# and have a kind of irrational exuberance, but is it actually irrational?
# I mean I'm not totally sure of what the mainstream economic thought on it is 
#actually, so I should look this up, but I mean in some ways, and I'm sure there is 
#irrationality, but in a lot of ways, contributing to the bubble if you are a short term
#speculator could basically be betting that you can buy and be up and sell before it bursts
# and over short time horizins you are almost certainly due to be correct
# so people betting that would bet up the price far above it's actual value
#since the actual value does not matter to them but rather the expected short term
#value gradient, and such a process could continue indefinitely, I suppose
# until either speculators run out of money, or there is some kind of exogenous
#shock that means the expeected value is down, so nobody would invest
#and everybody would sell, and that would also be self-sustaining
# and a crash would commence until below the vlalue
#when higher valued  time horizon people would buy it up again
# since it's below the actual realistic value
#and that is how it would work. I mean that's a fairly simple model
# and it could be fun to implement
#even though I'm doing phd procrastination'and to be better at julia
#I'm not sure this will even be that useful... dagnabbit

# so, how is the model actually going to work? obviously it will evolve in development
# but for an early start I was thinking about having a selection of agents
# with free money and goods
# they purchase goods from one another and attempt to increase their money
# by predicting prices, and they have different time horizons
# to their predictions - i.e. they look at the time average gradients of the 
# stock over a longer time period than before, and also look at averages and gradients
# over that time period
# and that they then use this informatoin, in a very simple and idealised ways
# to submit trades to a matching engine upon each time tick, which works
# like a stock market matching engine, so it's basically a virtual stock market
# and then this will track variou sthigns which could be useful
#but which I don't really understand
# so it could be interesting to try to figure out how this would work
#even if it is phd procrastination!

type Good:
	id::Integer,
	current_price::Integer,
	price_history::Array{Integer},
end

type Bought_Good:
	buy_price::Integer,
	quantity::Integer,
	tick_bought::Integer,
	good_id::Integer
end

type Transaction:
	good_id::Integer,
	quantity: Integer,
	type::Integer, #buy or sell
	price::Integer
end

type Offer:
	good_id::Integer,
	quantity::Integer,
	type::Integer,
	max_price::Integer,
	min_price::Integer
end

type Agent:
	curreny_money::Integer,
	goods_owned::Array{Bought_Good},
	transaction_history::Array{Transaction}
	time_horizon::Integer
end

type Match:
	buy_offer::Offer,
	sell_offer::Offer,
	match_price::Integer


type MatchingEngine:
	buy_offers: Array{Offer},
	sell_offers: Array{Offer},
	matches:: Array{Match}
end

# okay, that shuold set up the basics here, but I'm not sure what eles I need beyond this
#type wise, but the main thing is obviously not going to be the types but the actual
# functions which operate on the types, fairly obviously!

#so what is next. The matching algorithm is going to be kind of funny, I think
#for each buy offer, it traverses sell offers seeing if one matches
# and if it does, it creates a match, and removes the sell offers from the array
#and creates a match
# what about partial matches
# ack! that's goind to be horrible
# since you can construct it in a nasty way
# I think I'm just going to ignore that for now!
# who should be favoured in the price, or should it be split down the middle
#I think splitting down the middle would be reasonable!
function match!(engine::MatchingEngine):
	#just try this
	for buy_offer in engine.buy_offers:
		const good_id = buy_offer.good_id
		const quantity = buy_offer.quantity
		const max_price = buy_offer.max_price
		const min_price = buy_offer.min_price
		for sell_offer in engine.sell_offers:
			if sell_offer.good_id != good_id:
				break
			end
			if quantity !=sell_offer.quantity:
				break
			end
			if sell_offer.max_price < min_price:
				break
			end
			if sell_offer.min_price >max_price:
				break;
			end
			#so here there is a match. calculate the actual price
			#I can't focus










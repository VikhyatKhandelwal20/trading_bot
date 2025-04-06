import logging
import requests
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Dict, List

from hummingbot.core.data_type.common import OrderType, TradeType, PriceType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.event.events import OrderFilledEvent


class AdvancedMarketMaking(ScriptStrategyBase):
    """
    Advanced Market Making Strategy with:
    - NATR-based volatility adjustments
    - RSI/MACD trend analysis
    - Inventory skew management
    - Stop-loss mechanism
    - Fear & Greed sentiment integration
    """
    
    # Strategy Configuration
    bid_spread_scalar = 0.5
    ask_spread_scalar = 0.5
    candles_length = 10
    order_refresh_time = 30
    order_amount = 0.025
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    
   
    
    markets = {exchange: {trading_pair}}
    # Risk Management
    max_shift_spread = 0.005
    target_inventory_ratio = 0.5
    stop_loss_threshold = 0.10
    inventory_scalar = 0.5
    
    # Sentiment Analysis
    sentiment_api_url = "https://api.alternative.me/fng/?limit=1"
    sentiment_score = 50
    sentiment_spread_adjustment = 0.0005
    greed_threshold = 75
    fear_threshold = 25
    
    # Technical Indicators
    rsi_length = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    # Candles Configuration
    candle_exchange = "binance"
    candles_interval = "1m"
    max_records = 1000
    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        
        self.bid_spread_scalar = Decimal(str(self.bid_spread_scalar))
        self.ask_spread_scalar = Decimal(str(self.ask_spread_scalar))
        self.max_shift_spread = Decimal(str(self.max_shift_spread))
        self.target_inventory_ratio = Decimal(str(self.target_inventory_ratio))
        self.stop_loss_threshold = Decimal(str(self.stop_loss_threshold))
        self.inventory_scalar = Decimal(str(self.inventory_scalar))
        self.sentiment_spread_adjustment = Decimal(str(self.sentiment_spread_adjustment))
        self.greed_threshold = Decimal(str(self.greed_threshold))
        self.fear_threshold = Decimal(str(self.fear_threshold))
    



        self.create_timestamp = 0
        self.base, self.quote = self.trading_pair.split('-')
        self.candles = CandlesFactory.get_candle(CandlesConfig(
            connector=self.candle_exchange,
            trading_pair=self.trading_pair,
            interval=self.candles_interval,
            max_records=self.max_records
        ))
        self.candles.start()
        
        # Initialization
        self.reference_price = Decimal("0")
        self.entry_price = Decimal("0")
        self.buy_multiplier = Decimal("0")
        self.sell_multiplier = Decimal("0")
        self.price_multiplier = Decimal("0")
        self.bid_spread = Decimal("0.001")  
        self.ask_spread = Decimal("0.001")

    def on_stop(self):
        self.candles.stop()

    def on_tick(self):
        if self.current_timestamp >= self.create_timestamp:
            try:

                self.cancel_all_orders()
                self.update_multipliers()
                
                self.logger().info(f"Calculated Bid Spread: {self.bid_spread*100}%")
                self.logger().info(f"Calculated Ask Spread: {self.ask_spread*100}%")
            
                if self.check_stop_loss():
                    return
                    
                proposal = self.create_proposal()
                adjusted_proposal = self.adjust_proposal_to_budget(proposal)
                self.place_orders(adjusted_proposal)
                self.create_timestamp = self.current_timestamp + self.order_refresh_time

            except Exception as e:
                self.logger().error(f"Critical error in on_tick: {str(e)}", exc_info=True)
                self.notify_hb_app_with_timestamp(f"ERROR: {str(e)}")

    def update_multipliers(self):
        try:
            if len(self.candles.candles_df) < self.candles_length:
                self.logger().warning("Waiting for more candlestick data...")
                self.bid_spread = Decimal("0.001")
                self.ask_spread = Decimal("0.001")
                return

            # Get candlestick data
            candles_df = self.candles.candles_df

            natr_col = f"NATR_{self.candles_length}"
            rsi_col = f"RSI_{self.rsi_length}"
            macd_col = f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
            signal_col = f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"

            if natr_col not in candles_df.columns:
                candles_df.ta.natr(length=self.candles_length, append=True)
                
            if rsi_col not in candles_df.columns:
                candles_df.ta.rsi(length=self.rsi_length, append=True)
                
            if macd_col not in candles_df.columns or signal_col not in candles_df.columns:
                candles_df.ta.macd(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal, append=True)

            # Debug indicators
            self.logger().info(f"NATR column present: {natr_col in candles_df.columns}")
            if natr_col in candles_df.columns:
                self.logger().info(f"NATR values: {candles_df[natr_col].tail(1).values}")
            
            # Fetch sentiment
            self.fetch_sentiment()
            sentiment_effect = Decimal(str((self.sentiment_score - 50) / 250)) * self.sentiment_spread_adjustment
            sentiment_effect = max(Decimal("-0.0005"), min(Decimal("0.0005"), sentiment_effect))

            # NATR calculation with explicit type conversion
            if natr_col in candles_df.columns and not candles_df[natr_col].isna().all():
                natr_raw = candles_df[natr_col].iloc[-1]
                if not pd.isna(natr_raw):
                    # Conversion
                    natr_value = Decimal(str(natr_raw / 100))
                    self.logger().info(f"NATR calculation - raw: {natr_raw}, converted: {natr_value}")
                else:
                    self.logger().warning("NATR value is NaN, using default spread")
                    natr_value = Decimal("0.01") 
                    
            else:
                self.logger().warning(f"NATR column missing or all NaN, using default spread")
                natr_value = Decimal("0.01") 
               

            # Spreads
            self.bid_spread = max(
                Decimal('0.0003'), 
                min(Decimal('0.02'), natr_value * self.bid_spread_scalar + sentiment_effect)
            )

            self.ask_spread = max(
                Decimal('0.0003'), 
                min(Decimal('0.02'), natr_value * self.ask_spread_scalar - sentiment_effect)
            )

            # RSI and MACD 
            if rsi_col in candles_df.columns and macd_col in candles_df.columns and signal_col in candles_df.columns:
                if not pd.isna(candles_df[rsi_col].iloc[-1]) and not pd.isna(candles_df[macd_col].iloc[-1]) and not pd.isna(candles_df[signal_col].iloc[-1]):
                    rsi = candles_df[rsi_col].iloc[-1]
                    macd_diff = candles_df[macd_col].iloc[-1] - candles_df[signal_col].iloc[-1]
                    
                    # Conversion
                    rsi_effect = Decimal(str((rsi - 50) / 50)) * self.max_shift_spread
                    macd_effect = Decimal(str(macd_diff)) * self.max_shift_spread
                    
                    self.price_multiplier = max(
                        Decimal('-0.01'), 
                        min(Decimal('0.01'), rsi_effect + macd_effect)
                    )
                    self.logger().info(f"RSI: {rsi}, MACD diff: {macd_diff}, Multiplier: {self.price_multiplier}")
                else:
                    self.logger().warning("RSI or MACD contains NaN values, using default multiplier")
                    self.price_multiplier = Decimal('0')
            else:
                self.logger().warning("RSI or MACD columns missing, using default multiplier")
                self.price_multiplier = Decimal('0')

            # Market prices
            best_bid = self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.BestBid)
            best_ask = self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.BestAsk)
            
            # Conversion
            best_bid = Decimal(str(best_bid)) if best_bid is not None else None
            best_ask = Decimal(str(best_ask)) if best_ask is not None else None


           
            
            # Mid price
            if best_bid is not None and best_ask is not None and best_bid > 0 and best_ask > 0:
                mid_price = (best_bid + best_ask) / Decimal('2')
                self.logger().info(f"Mid price calculated: {mid_price}")
            else:
                self.logger().warning("Invalid bid/ask prices, trying last trade price")
                last_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.LastTrade)
                if last_price is not None and last_price > 0:
                    mid_price = Decimal(str(last_price))
                    self.logger().info(f"Using last trade price: {mid_price}")
                else:
                    self.logger().error("Cannot obtain valid price reference")
                    return

            


            # Volatility
            recent_prices = candles_df['close'].tail(5).values
            
            recent_prices_dec = [Decimal(str(price)) for price in recent_prices if not pd.isna(price)]

            
            if len(recent_prices_dec) >= 2 and mid_price > 0:
                price_range = (max(recent_prices_dec) - min(recent_prices_dec)) / mid_price
                
                if not isinstance(price_range, Decimal):
                    price_range = Decimal(str(price_range))
                
                if price_range > Decimal("0.005"):  
                    volatility_factor = min(Decimal("3"), (price_range * Decimal("100")))
                    self.bid_spread *= (Decimal("1") + (volatility_factor * Decimal("0.1")))
                    self.ask_spread *= (Decimal("1") + (volatility_factor * Decimal("0.1")))
                    self.logger().info(f"Increased spreads due to volatility: {volatility_factor}x")

            # Momentum
            try:
                
                short_term_return_float = candles_df['close'].pct_change(3).iloc[-1]
                
                
                if not pd.isna(short_term_return_float):
                    
                    short_term_return = Decimal(str(short_term_return_float))
                    momentum_factor = Decimal(str(abs(short_term_return_float) * 10))
                    
                    
                    if abs(short_term_return) > Decimal("0.005"):  
                        if short_term_return > 0:  # Upward momentum
                            # Widen buy spread (more conservative buys)
                            self.bid_spread *= (Decimal("1") + momentum_factor)
                            # sell spread tighten slightly
                            self.ask_spread *= Decimal("0.95")
                            self.logger().info(f"Upward momentum detected: {short_term_return}. Adjusting spreads.")
                        else:  # Downward momentum
                            # Widen sell spread (more conservative sells)
                            self.ask_spread *= (Decimal("1") + momentum_factor)
                            # buy spread tighten slightly
                            self.bid_spread *= Decimal("0.95")
                            self.logger().info(f"Downward momentum detected: {short_term_return}. Adjusting spreads.")
            except Exception as e:
                self.logger().warning(f"Momentum calculation failed: {str(e)}")
                            
            self.logger().info(f"Market prices - Best bid: {best_bid}, Best ask: {best_ask}")



            # Inventory skew
            base_bal = Decimal(str(self.connectors[self.exchange].get_balance(self.base)))
            quote_bal = Decimal(str(self.connectors[self.exchange].get_balance(self.quote)))
            
            self.logger().info(f"Balances - {self.base}: {base_bal}, {self.quote}: {quote_bal}")



            
            # Inventory ratio
            base_value = base_bal * mid_price
            total_value = base_value + quote_bal
            
            if total_value > 0:
                current_ratio = base_value / total_value
                inventory_delta = (self.target_inventory_ratio - current_ratio) / Decimal("0.5")
                self.logger().info(f"Inventory - Current ratio: {current_ratio}, Target: {self.target_inventory_ratio}, Delta: {inventory_delta}")
            else:
                self.logger().warning("Portfolio value is zero, using default ratio")
                current_ratio = Decimal('0.5')
                inventory_delta = Decimal('0')
            
            # Size multipliers
            self.buy_multiplier = max(
                Decimal('0.3'), 
                min(Decimal('2.0'), Decimal('1') + Decimal('3')*inventory_delta)
            )
            self.sell_multiplier = max(
                Decimal('0.3'), 
                min(Decimal('2.0'), Decimal('1') - Decimal('3')*inventory_delta)
            )
            
            # Sentiment
            if self.sentiment_score >= self.greed_threshold:
                self.sell_multiplier *= Decimal('1.2')
                self.buy_multiplier *= Decimal('0.8')
                self.logger().info("Sentiment adjustment: Greed - Favoring SELL")
            elif self.sentiment_score <= self.fear_threshold:
                self.buy_multiplier *= Decimal('1.2')
                self.sell_multiplier *= Decimal('0.8')
                self.logger().info("Sentiment adjustment: Fear - Favoring BUY")
            
            # Limits
            self.buy_multiplier = max(Decimal('0.4'), min(Decimal('1.6'), self.buy_multiplier))
            self.sell_multiplier = max(Decimal('0.4'), min(Decimal('1.6'), self.sell_multiplier))
            
            # Reference price
            self.reference_price = mid_price * (Decimal('1') + self.price_multiplier)
            self.logger().info(f"Final prices - Mid: {mid_price}, Reference: {self.reference_price}")
            
            
            if self.entry_price == Decimal('0'):
                self.entry_price = self.reference_price
                self.logger().info(f"Setting initial entry price: {self.entry_price}")

        except Exception as e:
            self.logger().error(f"Multiplier update failed: {str(e)}", exc_info=True)
            self.bid_spread = Decimal("0.001")
            self.ask_spread = Decimal("0.001")

    def fetch_sentiment(self):
        try:
            response = requests.get(self.sentiment_api_url, timeout=5)
            data = response.json()
            self.sentiment_score = int(data["data"][0]["value"])
        except Exception as e:
            self.logger().error(f"Failed to fetch sentiment: {str(e)}")
            self.sentiment_score = 50 

    def check_stop_loss(self):
        price_source = PriceType.MidPrice
        current_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, price_source)
        if current_price < self.entry_price * Decimal(1 - self.stop_loss_threshold):
            self.logger().info("Stop-loss triggered! Closing all positions.")
            self.cancel_all_orders()
            return True
        return False

    def create_proposal(self) -> List[OrderCandidate]:

        MIN_ORDER_AMOUNT = Decimal("0.01")
        
        
        self.logger().info(f"Creating proposal with reference price: {self.reference_price}")
        
    
        best_bid = self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.BestBid)
        best_ask = self.connectors[self.exchange].get_price_by_type(self.trading_pair, PriceType.BestAsk)
        self.logger().info(f"Proposal creation - Best bid: {best_bid}, Best ask: {best_ask}")
        
    
        buy_amount = max(MIN_ORDER_AMOUNT, Decimal(self.order_amount) * self.buy_multiplier)
        sell_amount = max(MIN_ORDER_AMOUNT, Decimal(self.order_amount) * self.sell_multiplier)
        
        if self.reference_price <= 0:
            self.logger().error("Reference price is zero or negative, cannot create valid proposal")
            return []

        fee_cost = Decimal("0.001")
        min_profit = Decimal("0.0005")
        
        
        # Prices
        buy_price_raw= self.reference_price * (1 - Decimal(self.bid_spread) - fee_cost - min_profit)
        sell_price_raw = self.reference_price * (1 + Decimal(self.ask_spread) + fee_cost + min_profit)

        buy_price = min(buy_price_raw, best_bid * Decimal("1.0005"))
        sell_price = max(sell_price_raw, best_ask * Decimal("0.9995"))
        
       
        
        self.logger().info(f"""
        Order Creation Details:
        - Reference Price: {self.reference_price:.4f}
        - Buy price (raw): {buy_price_raw:.4f}, Final: {buy_price:.4f}
        - Sell price (raw): {sell_price_raw:.4f}, Final: {sell_price:.4f}
        - Buy amount: {buy_amount}, Sell amount: {sell_amount}
        """)
        
        
        if buy_price <= 0 or sell_price <= 0:
            self.logger().error(f"Invalid prices calculated: buy={buy_price}, sell={sell_price}")
            return []
        
        return [
            OrderCandidate(
                trading_pair=self.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=buy_amount,
                price=buy_price
            ),
            OrderCandidate(
                trading_pair=self.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=sell_amount,
                price=sell_price
            )
        ]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        return self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)

    def place_orders(self, proposal: List[OrderCandidate]):
        for order in proposal:
            if order.order_side == TradeType.BUY:
                self.buy(
                    connector_name=self.exchange,
                    trading_pair=order.trading_pair,
                    amount=order.amount,
                    price=order.price,
                    order_type=order.order_type
                )
            else:
                self.sell(
                    connector_name=self.exchange,
                    trading_pair=order.trading_pair,
                    amount=order.amount,
                    price=order.price,
                    order_type=order.order_type
                )

    def cancel_all_orders(self):
        """Cancel all active orders"""
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(
                connector_name=self.exchange,
                trading_pair=order.trading_pair,
                order_id=order.client_order_id
            )

    def did_fill_order(self, event: OrderFilledEvent):
        self.logger().info(f"Filled {event.trade_type.name} {event.amount} @ {event.price}")
        self.entry_price = event.price  


      

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        
        lines = []
        lines.extend(["", "  Balances:"] + 
            [f"    {self.base}: {self.connectors[self.exchange].get_balance(self.base)}"] +
            [f"    {self.quote}: {self.connectors[self.exchange].get_balance(self.quote)}"])
        
        lines.extend(["", "  Current Spreads:"] +
            [f"    Bid: {self.bid_spread*10000:.2f} bps | Ask: {self.ask_spread*10000:.2f} bps"])
        
        lines.extend(["", "  Market Sentiment:"] +
            [f"    Fear & Greed Index: {self.sentiment_score}/100"])
        
        lines.extend(["", "  Risk Management:"] +
            [f"    Stop-loss: {self.stop_loss_threshold*100}% below entry price"] +
            [f"    Inventory Target: {self.target_inventory_ratio*100}% {self.base}"])
        
        return "\n".join(lines)
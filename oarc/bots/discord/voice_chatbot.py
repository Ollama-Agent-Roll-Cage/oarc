import discord
from discord.ext import commands

# Intents are required for certain events
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.voice_states = True

# Bot setup
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} and ready to go!")

# Join voice channel command
@bot.command()
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
        await ctx.send(f"Joined {channel.name}!")
    else:
        await ctx.send("You need to be in a voice channel first!")

# Leave voice channel command
@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("Disconnected from the voice channel!")
    else:
        await ctx.send("I'm not in a voice channel!")

# Run the bot
bot.run("YOUR_DISCORD_BOT_TOKEN")